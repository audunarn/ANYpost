"""Atomic HDF5 publication and selective, bounded offline replay."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from functools import wraps
from concurrent.futures import CancelledError
from pathlib import Path
from threading import RLock
import importlib
import hashlib
import json
import os
import tempfile
import numpy as np
import h5py
from .shared_metadata import CaseMetadataWriter, CaseMetadataReader, load_native_metadata
from . import storage_prediction as _prediction
from .cache import ByteLRU
from .codec import response_dataset, response_shape
from .stored_codec import (STORED_CODEC, stored_shape, stored_encoding_option, encode_tile_rows, write_corrections, decode_stored, correction_dataset, NATIVE_STORED_CODEC, plain_stored_dataset)
from .errors import (ValidationError, IdentityError, FailedCaseError, UnavailableFieldError,
                     RuntimeDependencyError, WorkingSetError, ClosedReaderError, UnknownCaseError, RecoveryError, RequestCancelled)
from .validation import validate_capsule, validate_metadata, json_text, SUCCESS

CHUNK_BYTES = 256 * 1024
COMPUTE_BYTES = 64 * 1024 * 1024

def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)

def _freeze(value):
    value.setflags(write=False)
    return value

def _selection_identity(indices):
    """Disjoint cache-key kinds preserve logical order and duplicates."""
    if isinstance(indices,range):
        return ("range",indices.start,indices.stop,indices.step)
    if len(indices)<16:
        return ("ids",tuple(int(index) for index in indices))
    return ("ids_hash",len(indices),hashlib.sha256(np.asarray(indices,dtype=np.int64).tobytes()).digest())

def _dataset(group, name, values, compression, *, integer=False):
    a = np.asarray(values)
    dtype = np.dtype("bool" if a.dtype.kind == "b" else "int64" if integer else "float64")
    kwargs = {}
    if a.size:
        shape = list(a.shape)
        while np.prod(shape, dtype=object) * 8 > CHUNK_BYTES:
            axis = max(range(len(shape)), key=lambda i: shape[i])
            shape[axis] = max(1, (shape[axis] + 1) // 2)
        kwargs = dict(chunks=tuple(shape), compression=compression, shuffle=True, fletcher32=True)
    ds = group.create_dataset(name, shape=a.shape, dtype=dtype, **kwargs)
    if a.ndim == 0:
        ds[()] = a
    elif a.size:
        # Rows alone can exceed chunk target: iterate every HDF5 chunk.
        for block in ds.iter_chunks():
            ds[block] = a[block]
    return ds


class _AssociationPool:
    """Transaction-local canonical tables; hardlink datasets, never groups.

    Only association paths participate. Geometry IDs have separate semantics.
    Hashing and collision verification use bounded blocks and retain no arrays.
    """
    def __init__(self):
        self._tables = {}

    def __call__(self, group, name, values, compression, *, integer=False):
        if name != "associations" or not integer:
            return _dataset(group, name, values, compression, integer=integer)
        a = np.asarray(values)
        if a.ndim not in (1, 2) or a.dtype.kind not in "iu":
            raise ValidationError("association pool requires a one/two-dimensional integer table")
        digest = hashlib.sha256()
        # Canonical int64 matches the persisted dtype; order and shape matter.
        with np.nditer(a, flags=["external_loop", "buffered", "zerosize_ok"],
                       op_flags=["readonly"], op_dtypes=[np.dtype("int64")],
                       casting="unsafe", order="C", buffersize=CHUNK_BYTES//8) as chunks:
            for block in chunks:
                digest.update(block.tobytes())
        key = (a.shape, digest.digest())
        prior = self._tables.get(key)
        if prior is not None:
            if a.size:
                for block in prior.iter_chunks():
                    if not np.array_equal(prior[block], np.asarray(a[block], dtype=np.int64)):
                        raise ValidationError("association content digest collision")
            group[name] = prior
            return group[name]
        dataset = _dataset(group, name, a, compression, integer=True)
        self._tables[key] = dataset
        return dataset


def _writer_cancellation(cancel_event, cancel_check):
    # Validate exclusivity/types before creating directories or temporary files.
    if cancel_event is not None and cancel_check is not None:
        raise TypeError("supply cancel_event or cancel_check, not both")
    if cancel_event is not None and not callable(getattr(cancel_event, "is_set", None)):
        raise TypeError("cancel_event must expose is_set()")
    if cancel_check is not None and not callable(cancel_check):
        raise TypeError("cancel_check must be callable")
    def check():
        if cancel_event is not None and cancel_event.is_set():
            raise RequestCancelled("archive write cancelled")
        if cancel_check is not None:
            cancel_check()  # Owner exceptions (including SolveCancelled) pass through.
    return check


def write_results(path, capsule, *, compression="lzf", stored_encoding="auto", cancel_event=None, cancel_check=None):
    """Validate then atomically replace destination; never publish partial data."""
    check = _writer_cancellation(cancel_event, cancel_check)
    check()
    if not isinstance(capsule, Mapping):
        raise ValidationError("expected recovery capsule mapping")
    if compression not in ("lzf", None):
        raise ValueError("compression must be 'lzf' or None")
    stored_encoding_option(stored_encoding)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, "w") as handle:
            peak_encode_work_bytes = 0
            def observe_work(nbytes):
                nonlocal peak_encode_work_bytes
                peak_encode_work_bytes = max(peak_encode_work_bytes, nbytes)
            pool = CaseMetadataWriter(handle, capsule.get("model_identity"), capsule.get("mesh_identity"), compression)
            compact_capsule = dict(capsule, cases=[pool.add_case(case,index) for index,case in enumerate(capsule.get("cases",()))])
            meta = validate_capsule({k:v for k,v in compact_capsule.items() if k != "storage_prediction"})
            handle.attrs.update(schema="anypost.results", version=1, complete=False,
                                runtime=meta["runtime"], model_identity=meta["model_identity"],
                                mesh_identity=meta["mesh_identity"])
            metadata = dict(meta)
            metadata["source_metadata"] = {k: v for k, v in capsule.items() if k not in {"schema", "version", "runtime", "model_identity", "mesh_identity", "solver_identity", "geometry", "patterns", "coefficients", "cases", "descriptors", "associations", "responses", "stored", "derived", "storage_prediction"}}
            metadata["component_groups"] = {d["key"]: str(i) for i, d in enumerate(meta["descriptors"])}
            metadata["stored_case_indices"] = {k: np.asarray(v["case_indices"]).tolist() for k, v in capsule["stored"].items()}
            role=capsule.get("storage_prediction")
            if role is not None:
                if meta["capsule_version"]!=2:raise ValidationError("storage prediction requires capsule2")
                metadata["storage_prediction"]=_prediction.manifest(role,meta["patterns"],capsule["associations"],meta["descriptors"])
            handle.attrs["version"] = pool.finish(metadata)
            handle.create_dataset("metadata_json", data=json_text(metadata), dtype=h5py.string_dtype("utf-8"))
            _dataset(handle, "coefficients", capsule["coefficients"], compression)
            geometry = handle.create_group("geometry")
            for name, value in capsule["geometry"].items():
                _dataset(geometry, name, value, compression, integer=np.asarray(value).dtype.kind in "iu")
            components = handle.create_group("components")
            association_pool = _AssociationPool()
            if role is not None:
                _prediction.install(handle,role,meta["patterns"],lambda k:metadata["component_groups"][k],association_pool,compression)
            for key, name in metadata["component_groups"].items():
                check()
                group = components.create_group(name)
                if key in capsule["associations"]:
                    association_pool(group, "associations", capsule["associations"][key], compression, integer=True)
                if key in capsule["responses"]:
                    response_dataset(group, capsule["responses"][key], capsule["associations"].get(key), compression, _dataset)
                if key in capsule["stored"]:
                    entry = capsule["stored"][key]
                    _dataset(group, "case_indices", entry["case_indices"], compression, integer=True)
                    values = np.asarray(entry["values"])
                    if stored_encoding == "auto" and role is not None and key in role["targets"] and values.size:
                        ds=_prediction.create_values(group,len(values),values.shape[1],compression,info=metadata["storage_prediction"],key=key,width=len(meta["patterns"])+1)
                        _prediction.write_values(handle,group,metadata["storage_prediction"],key,values,entry["case_indices"],handle["coefficients"],meta["cases"],metadata["component_groups"],check=check,on_work=observe_work)
                    elif stored_encoding == "auto" and "responses" in group and values.size:
                        ds = correction_dataset(group,len(values),values.shape[1],group["responses"],compression)
                        write_corrections(ds, values, entry["case_indices"], group["responses"], capsule["coefficients"], meta["cases"],on_work=observe_work)
                    else:
                        plain_stored_dataset(group,values,compression)
            if role is not None:
                _prediction.validate_disk(handle,metadata["storage_prediction"],meta["patterns"],metadata["component_groups"],{d["key"]:d for d in meta["descriptors"]},{k:len(v) for k,v in capsule["associations"].items()})
            check()
            handle.attrs["peak_encode_work_bytes"] = peak_encode_work_bytes
            handle.attrs["complete"] = True
            handle.flush()
        with open(temporary, "r+b") as stream:
            os.fsync(stream.fileno())
        check()
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path

@dataclass(frozen=True)
class Envelope:
    minimum: np.ndarray
    maximum: np.ndarray
    min_case_ids: np.ndarray
    max_case_ids: np.ndarray
    admitted_case_ids: tuple[str, ...] = ()
    failed_case_ids: tuple[str, ...] = ()
    requested_case_ids: tuple[str, ...] = ()
    @property
    def excluded_case_ids(self): return self.failed_case_ids
    @property
    def min(self): return self.minimum
    @property
    def max(self): return self.maximum
    @property
    def governing_min(self): return self.min_case_ids
    @property
    def governing_max(self): return self.max_case_ids
    def __getitem__(self, key): return getattr(self, key)

def _io_boundary(method):
    """Translate HDF5 transport/checksum failures without losing the cause."""
    @wraps(method)
    def typed(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except OSError as exc:
            raise ValidationError(f"HDF5 {method.__name__} failed: {exc}") from exc
    return typed

def _cancellable(method):
    @wraps(method)
    def cancellable(self, *args, **kwargs):
        token = kwargs.get("cancel_event")
        if token is not None and (not callable(getattr(token,"is_set",None))):
            raise TypeError("cancel_event must expose is_set()")
        if token is not None and token.is_set(): raise CancelledError()
        with self._lock:
            previous = getattr(self,"_cancel_event",None)
            self._cancel_event = token if token is not None else previous
            try:
                self._check_cancel()
                result = method(self,*args,**kwargs)
                self._check_cancel()
                return result
            finally:
                self._cancel_event = previous
    return cancellable

class ResultsReader:
    supports_cancellation = True
    def _check_cancel(self):
        token = getattr(self,"_cancel_event",None)
        if token is not None and token.is_set(): raise CancelledError()
    def _cache_put(self,key,value):
        self._check_cancel()
        return self._cache.put(key,value)
    @_io_boundary
    def __init__(self, path, *, cache_bytes=2*1024**3, working_bytes=256*1024**2,
                 expected_model_identity=None, expected_mesh_identity=None, derivation_workers=None):
        if isinstance(working_bytes, bool) or not isinstance(working_bytes, int) or working_bytes <= 0:
            raise ValueError("working_bytes must be a positive integer")
        if derivation_workers is not None and (isinstance(derivation_workers,bool) or not isinstance(derivation_workers,int) or derivation_workers not in (1,2)):
            raise ValueError("derivation_workers must be None, 1 or 2")
        self.derivation_workers=derivation_workers
        self._derivation_effective_workers=1
        self._derivation_parallel_calls=0
        self._derivation_calls=0
        self.path = Path(path)
        self.working_bytes = working_bytes
        self._cache = ByteLRU(cache_bytes)
        self._lock = RLock()
        self._closed = False
        self._reads = self._reconstructions = self._working_peak = 0
        self._coefficient_reads = self._coefficient_hits = 0
        self._association_reads = self._association_hits = self._association_verifications = 0
        self._verified_associations = set()
        self._compute_tiles = self._largest_compute_tile = 0
        self._runtime = None
        self._component_handles = {}
        self._dataset_handles = {}
        self._response_keys = set()
        self._response_layouts = {}
        self._stored_codecs = set()
        self._prediction_codecs=set()
        self._prediction_info=None
        self._prediction_tile_reads=0
        self._prediction_basis_reads=self._prediction_basis_hits=0
        self._stored_decodes = 0
        self._verified_codecs = set()
        self._prepared_response_reads = self._prepared_response_hits = self._prepared_reconstructions = 0
        try:
            self._handle = h5py.File(path, "r", rdcc_nbytes=0)
            self._load_metadata()
            if not hasattr(self,"_case_metadata"):
                # Legacy subclasses retain their existing header/record meaning.
                self._case_metadata = CaseMetadataReader(None,self._meta,1,self.working_bytes)
            for expected, actual, label in ((expected_model_identity, self.model_identity, "model"),
                                            (expected_mesh_identity, self.mesh_identity, "mesh")):
                if expected is not None and expected != actual:
                    raise IdentityError(f"{label} identity mismatch")
        except Exception:
            if hasattr(self, "_handle"):
                self._handle.close()
            raise

    def _load_metadata(self):
        h = self._handle
        seen_groups = set()
        def check_links(group):
            address = h5py.h5o.get_info(group.id).addr
            if address in seen_groups:
                raise ValidationError("cyclic or aliased HDF5 groups are not supported")
            seen_groups.add(address)
            for name in group:
                link = group.get(name, getlink=True)
                if not isinstance(link, h5py.HardLink):
                    raise ValidationError("external and soft HDF5 links are not supported")
                item = group[name]
                if isinstance(item, h5py.Group):
                    check_links(item)
                elif isinstance(item, h5py.Dataset) and (item.is_virtual or item.external):
                    raise ValidationError("native datasets must be self-contained, without virtual or external storage")
        check_links(h)
        native_version=h.attrs.get("version")
        if _text(h.attrs.get("schema", "")) != "anypost.results" or isinstance(native_version,(bool,np.bool_)) or not isinstance(native_version,(int,np.integer)) or native_version not in (1,2) or not h.attrs.get("complete", False):
            raise ValidationError("unsupported or incomplete native archive")
        try:
            self._meta = load_native_metadata(_text(h["metadata_json"][()]),int(h.attrs["version"]))
            validate_metadata(self._meta)
            self._case_metadata = CaseMetadataReader(h,self._meta,int(h.attrs["version"]),self.working_bytes)
            for name in ("runtime", "model_identity", "mesh_identity"):
                if _text(h.attrs.get(name, "")) != self._meta[name]:
                    raise ValidationError(f"{name} disagrees with metadata")
            self.model_identity = self._meta["model_identity"]
            self.mesh_identity = self._meta["mesh_identity"]
            self._prediction_info=self._meta.get("storage_prediction")
            self._prediction_identity=hashlib.sha256(json.dumps(self._prediction_info,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            self._descriptors = {d["key"]: d for d in self._meta["descriptors"]}
            self._pairs = {(d["quantity_id"], d["component"]): d["key"] for d in self._meta["descriptors"]}
            self._case_index = {c["case_id"]: i for i, c in enumerate(self._meta["cases"])}
            self._groups = self._meta["component_groups"]
            if set(self._groups) != set(self._descriptors) or len(set(self._groups.values())) != len(self._groups) or any(not isinstance(v, str) or not v.isdigit() for v in self._groups.values()):
                raise ValidationError("invalid component group metadata")
            coefficients = h["coefficients"]
            if coefficients.shape != (len(self._case_index), len(self._meta["patterns"])) or coefficients.dtype != np.dtype("float64"):
                raise ValidationError("coefficient dataset mismatch")
            if self._meta.get("capsule_version",1) == 2: self._validate_geometry_maps()
            self._counts, self._stored_rows = {}, {}
            for key, descriptor in self._descriptors.items():
                group = self._group(key)
                assoc = group.get("associations")
                count = 0 if assoc is None else assoc.shape[0]
                self._counts[key] = count
                if assoc is not None and (assoc.ndim not in (1, 2) or assoc.dtype.kind not in "iu"):
                    raise ValidationError("invalid association dataset")
                if assoc is None and descriptor["availability"] != "unavailable":
                    raise ValidationError("missing field association dataset")
                if "responses" in group:
                    self._response_keys.add(key)
                if "responses" in group:
                    ds = group["responses"]
                    if ds.dtype != np.dtype("float64"):
                        raise ValidationError("invalid response dataset")
                    self._response_layouts[key] = response_shape(ds, count, len(self._meta["patterns"])+1)
                    if self._response_layouts[key] is not None and (assoc is None or (assoc.ndim == 2 and assoc.shape[1] == 0)):
                        raise ValidationError("uniform sample codec requires leading association IDs")
                ids = self._meta.get("stored_case_indices", {}).get(key, [])
                if any(type(i) is not int or i < 0 or i >= len(self._case_index) for i in ids) or len(set(ids)) != len(ids):
                    raise ValidationError("invalid explicit row metadata")
                self._stored_rows[key] = {i: row for row, i in enumerate(ids)}
                if "values" in group:
                    if "case_indices" not in group or group["case_indices"].shape != (len(ids),):
                        raise ValidationError("invalid explicit value dataset")
                    if group["values"].attrs.get("codec")==_prediction.CODEC:
                        if self._prediction_info is None or key not in self._prediction_info.get("targets",()):raise ValidationError("storage codec has no declared prediction target")
                        _prediction.validate_values(group,len(ids),count)
                        self._prediction_codecs.add(key)
                    elif stored_shape(group["values"], (len(ids), count), key in self._response_keys,response=group.get("responses")):
                        self._stored_codecs.add(key)
                if key not in self._prediction_codecs and ("prediction_modes" in group or "decoded_sha256" in group):
                    raise ValidationError("stray storage prediction integrity datasets")
                if "case_indices" in group:
                    recorded = group["case_indices"]
                    if recorded.dtype.kind not in "iu" or recorded.shape != (len(ids),) or not np.array_equal(recorded[()], ids):
                        raise ValidationError("explicit row dataset disagrees with metadata")
                if ids and "values" not in group:
                    raise ValidationError("missing explicit value dataset")
                availability = descriptor["availability"]
                if availability == "unavailable" and ("values" in group or "responses" in group or key in self._meta["derived"]):
                    raise ValidationError("unavailable field contains replay data")
                if availability == "stored" and "values" not in group:
                    raise ValidationError("missing stored field data")
                if availability == "reconstructable" and "responses" not in group and key not in self._meta["derived"]:
                    raise ValidationError("missing field replay path")
                for dep in self._meta["derived"].get(key, {}).get("inputs", {}).values():
                    if self._group(dep)["associations"].shape != assoc.shape:
                        raise ValidationError("derived association shapes differ")
            _prediction.validate_disk(h,self._prediction_info,self._meta["patterns"],self._groups,self._descriptors,self._counts,self._check_cancel,self._reserve)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            if isinstance(exc, ValidationError): raise
            raise ValidationError(f"malformed archive metadata: {exc}") from exc

    def _validate_geometry_maps(self):
        geometry=self._handle["geometry"]
        positions=geometry.get("positions")
        if positions is not None and (positions.ndim!=2 or positions.shape[1]!=3 or positions.dtype.kind not in "fiu"):
            raise ValidationError("geometry positions require numeric (count,3)")
        node_ids=geometry.get("node_ids")
        if node_ids is not None and (positions is None or node_ids.ndim!=1 or node_ids.dtype.kind not in "iu" or len(node_ids)!=len(positions)):
            raise ValidationError("geometry node IDs disagree with positions")
        for name,width in (("triangles",3),("lines",2),("point_indices",None)):
            ds=geometry.get(name)
            if ds is None: continue
            if positions is None or ds.dtype.kind not in "iu" or (ds.ndim!=1 if width is None else ds.ndim!=2 or ds.shape[1]!=width):
                raise ValidationError("geometry topology requires integer indices and positions")
            step=max(1,min(32768,self.working_bytes//max(32,(width or 1)*16)))
            for first in range(0,len(ds),step):
                stop=min(first+step,len(ds));self._reserve((stop-first)*(width or 1)*16)
                block=ds[first:stop]
                if np.any(block<0) or np.any(block>=len(positions)):raise ValidationError("geometry point index outside positions")
        element_ids=geometry.get("element_ids")
        if element_ids is not None and (element_ids.ndim!=1 or element_ids.dtype.kind not in "iu"):
            raise ValidationError("geometry element IDs require one integer axis")
        for name,topology in (("line_to_element","lines"),("triangle_to_element","triangles")):
            ds=geometry.get(name)
            if ds is None: continue
            if element_ids is None or topology not in geometry or ds.ndim!=1 or ds.dtype.kind not in "iu" or len(ds)!=len(geometry[topology]):
                raise ValidationError("geometry dense mapping requires element IDs and matching topology")
            step=max(1,min(32768,self.working_bytes//32))
            for first in range(0,len(ds),step):
                stop=min(first+step,len(ds));self._reserve((stop-first)*32)
                block=ds[first:stop]
                if np.any(block<0) or np.any(block>=len(element_ids)):raise ValidationError("geometry dense element index outside element IDs")

    def _group(self, key):
        group = self._component_handles.get(key)
        if group is None:
            group = self._handle["components"][self._groups[key]]
            self._component_handles[key] = group
        return group
    def _component_dataset(self, key, name):
        identity = (key, name)
        dataset = self._dataset_handles.get(identity)
        if dataset is None:
            dataset = self._group(key)[name]
            self._dataset_handles[identity] = dataset
        return dataset
    def _ensure_open(self):
        if self._closed: raise ClosedReaderError("result reader is closed")
    def __enter__(self):
        self._ensure_open(); return self
    def __exit__(self, *args): self.close()
    def close(self):
        with self._lock:
            if not self._closed:
                self._handle.close(); self._cache.clear(); self._cache._external = {}; self._component_handles.clear(); self._dataset_handles.clear(); self._closed = True
    @_io_boundary
    def cases(self):
        with self._lock:
            self._ensure_open()
            self._case_metadata.require_detail_work(range(len(self._meta["cases"])))
            return tuple(self._case_metadata.detail(index) for index in range(len(self._meta["cases"])))
    def components(self, quantity_id=None):
        with self._lock:
            self._ensure_open()
            return tuple(json.loads(json_text(d)) for d in self._meta["descriptors"] if quantity_id is None or d["quantity_id"] == quantity_id)
    @_io_boundary
    def metadata(self):
        with self._lock:
            self._ensure_open(); return self._case_metadata.public_metadata(details=True)
    @_io_boundary
    def case_metadata(self, case_id):
        with self._lock:
            self._ensure_open()
            try: index=self._case_index[case_id]
            except (KeyError,TypeError):raise UnknownCaseError(f"unknown case {case_id!r}") from None
            return self._case_metadata.detail(index)
    def case_summaries(self):
        with self._lock:
            self._ensure_open(); return self._case_metadata.summaries()
    def metadata_summary(self):
        with self._lock:
            self._ensure_open(); return self._case_metadata.public_metadata(details=False)
    @property
    def diagnostics(self):
        external = sum(getattr(self._cache, "_external", {}).values())
        runtime=self._meta.get("runtime")
        capsule_version=self._meta.get("capsule_version",1 if runtime=="anysolver-results/1" else None)
        combination=self._meta.get("combination","legacy_blas/1" if runtime=="anysolver-results/1" else None)
        return dict(capsule_version=capsule_version,runtime=runtime,combination=combination,
                    response_layout_deterministic=True if combination=="ordered_float64/1" else False if combination=="legacy_blas/1" else None,
                    determinism_scope="response_combination_only",
                    derivation_workers_requested=self.derivation_workers,
                    derivation_workers_effective=self._derivation_effective_workers,
                    derivation_parallel_calls=self._derivation_parallel_calls,
                    derivation_calls=self._derivation_calls,
                    cache_resident_bytes=self._cache.resident_bytes+external,
                    cache_peak_bytes=self._cache.peak_bytes, external_cache_bytes=external,
                    cache_hits=self._cache.hits, cache_misses=self._cache.misses,
                    reads=self._reads, reconstructions=self._reconstructions,
                    coefficient_reads=self._coefficient_reads, coefficient_hits=self._coefficient_hits,
                    association_reads=self._association_reads, association_hits=self._association_hits,
                    association_verifications=self._association_verifications,
                    verified_association_edges=len(self._verified_associations),
                    compute_tiles=self._compute_tiles, largest_compute_tile=self._largest_compute_tile,
                    stored_correction_decodes=getattr(self, "_stored_decodes", 0),
                    storage_prediction_tile_decodes=getattr(self,"_prediction_tile_reads",0),
                    storage_prediction_basis_reads=self._prediction_basis_reads,storage_prediction_basis_hits=self._prediction_basis_hits,
                    prepared_response_creations=self._prepared_response_reads,
                    prepared_response_hits=self._prepared_response_hits,
                    prepared_reconstructions=self._prepared_reconstructions,
                    working_peak_bytes=self._working_peak, cache_bytes=self._cache.capacity,
                    working_bytes=self.working_bytes,
                    shared_case_tables=len(self._case_metadata.tables),
                    shared_case_references=len(self._case_metadata.references),
                    shared_metadata_verified_bytes=self._case_metadata.verified_table_bytes)
    def set_external_cache_bytes(self, owner_key, byte_count):
        with self._lock:
            self._ensure_open(); self._cache.set_external_bytes(owner_key, byte_count)
    def clear_cache(self):
        """Clear evictable fields/basis; external live buffers retain reservations."""
        with self._lock:
            self._ensure_open(); self._cache.clear()
    def _reserve(self, amount):
        self._check_cancel()
        if amount > self.working_bytes:
            raise WorkingSetError(f"request needs at least {amount} bytes; working cap is {self.working_bytes}")
        self._working_peak = max(self._working_peak, amount)
    def _key(self, quantity, component):
        try: return self._pairs[(quantity, component)]
        except KeyError: raise UnavailableFieldError(f"unknown quantity/component {quantity!r}/{component!r}") from None
    def _selection(self, count, selection):
        if selection is None: return range(count), ("all",)
        if isinstance(selection, slice):
            start, stop, step = selection.indices(count)
            return range(start, stop, step), ("slice", start, stop, step)
        if hasattr(selection, "__len__"):
            self._reserve(len(selection) * 16)
        a = np.asarray(selection)
        if a.ndim == 0: a = a.reshape(1)
        if a.size == 0 and a.ndim == 1: a = np.empty(0, dtype=np.int64)
        if a.ndim != 1 or a.dtype.kind not in "iu":
            raise ValidationError("selection must be integer flattened indices or slice")
        self._reserve(a.size * 16)
        if np.any(a < 0) or np.any(a >= count): raise IndexError("selection out of bounds")
        a = np.asarray(a, dtype=np.int64)
        # Bytes preserve order and duplicates; no full-array log is emitted.
        return a, ("indices", hashlib.sha256(a.tobytes()).digest())
    def _runtime_module(self):
        if self._runtime is None:
            try: runtime = importlib.import_module("anysolver_results")
            except ImportError as exc:
                raise RuntimeDependencyError("recovery needs optional ANYsolver-results runtime") from exc
            requested = self._meta["runtime"]
            identity = getattr(runtime,"RUNTIME_ID",None)
            if identity != requested and requested not in getattr(runtime,"SUPPORTED_RUNTIME_IDS",()):
                raise RuntimeDependencyError("lightweight runtime identity mismatch")
            if requested == "anysolver-results/2" and (identity != requested or requested not in getattr(runtime,"SUPPORTED_RUNTIME_IDS",()) or self._meta["combination"] not in getattr(runtime,"SUPPORTED_COMBINATIONS",())):
                raise RuntimeDependencyError("lightweight runtime lacks ordered recovery capability")
            self._runtime = runtime
        return self._runtime
    def _runtime_kwargs(self):
        return {"combination":self._meta.get("combination","legacy_blas/1")} if getattr(self._runtime_module(),"RUNTIME_ID",None)=="anysolver-results/2" else {}
    def _runtime_call(self, case_index, key, function, *args, **kwargs):
        runtime=self._runtime_module()
        errors=tuple(error for name in ("ReconstructionError","DerivationError")
                     if isinstance(error:=getattr(runtime,name,None),type) and issubclass(error,Exception))
        self._check_cancel()
        try:
            value=function(*args,**kwargs)
        except errors as exc:
            case_id=self._meta["cases"][case_index]["case_id"]
            raise RecoveryError(f"case {case_id!r}, component {key!r}: {exc}",
                                case_id=case_id,component_key=key,runtime=self._meta["runtime"],
                                combination=self._meta.get("combination","legacy_blas/1")) from exc
        self._check_cancel()
        return value
    def _admit(self, case_index, key, seen=None, *, runtime_required=True):
        """Refuse source-unavailable paths before allocating even empty outputs."""
        self._check_cancel()
        seen = set() if seen is None else seen
        if key in seen: return
        seen.add(key)
        case = self._meta["cases"][case_index]
        if key in case.get("unavailable_components",()) or self._descriptors[key]["availability"] == "unavailable":
            raise UnavailableFieldError(f"{key}: unavailable for case {case['case_id']!r}")
        if case_index in self._stored_rows[key]: return
        if self._descriptors[key]["availability"] == "stored":
            raise UnavailableFieldError(f"{key}: no explicit stored row for this case")
        if case.get("portable") is False or (self._meta.get("capsule_version",1)==2 and case.get("portable") is not True):
            raise UnavailableFieldError(f"{key}: source case does not permit response replay")
        if (key.startswith("displacement:") or self._descriptors[key]["quantity_id"] in ("displacement","displacements")) and "displacement_portable" in case and case["displacement_portable"] is not True:
            raise UnavailableFieldError(f"{key}: source case does not permit displacement replay")
        if not case["certified"]:
            raise UnavailableFieldError("uncertified case has no explicit stored field")
        for dep in self._meta["derived"].get(key, {}).get("inputs", {}).values():
            self._admit(case_index, dep, seen, runtime_required=runtime_required)
        if runtime_required: self._runtime_module()
    def case_components(self, case_id, quantity_id=None):
        """Discover case-specific routes without field reads or runtime imports."""
        with self._lock:
            self._ensure_open(); index,_ = self._case(case_id)
            rows=[]
            for d in self.components(quantity_id):
                try:
                    self._admit(index,d["key"],runtime_required=False)
                    d["availability"]="stored" if d["availability"]=="stored" or index in getattr(self,"_stored_rows",{}).get(d["key"],{}) else "reconstructable"
                except UnavailableFieldError:
                    d["availability"]="unavailable"
                rows.append(d)
            return tuple(rows)
    def _case(self, case_id):
        self._check_cancel()
        try: index = self._case_index[case_id]
        except (KeyError,TypeError): raise UnknownCaseError(f"unknown case {case_id!r}") from None
        case = self._meta["cases"][index]
        if case["status"] not in SUCCESS: raise FailedCaseError(f"case {case_id!r}: {case['status']}; see cases() diagnostics")
        return index, case
    @_io_boundary
    def _spatial_read(self, dataset, indices, row=None):
        self._check_cancel()
        self._reads += 1
        if dataset.ndim == 3:
            return self._uniform_read(dataset, indices) if row is None else self._native_stored_read(dataset,indices,row)
        if not len(indices): return np.empty((0,) + dataset.shape[(1 if row is None else 2):], dtype=dataset.dtype)
        # Regular positive strides avoid fancy-index bookkeeping. Arbitrary order uses
        # sorted unique indices only within a bounded spatial block.
        sl = None
        ids = None
        if isinstance(indices, range) and indices.step > 0:
            sl = slice(indices.start, indices.stop, indices.step)
        else:
            ids = np.asarray(indices, dtype=np.int64)
            if len(ids) == 1:
                sl = slice(int(ids[0]), int(ids[0])+1, 1)
            elif len(ids) > 1:
                step = int(ids[1])-int(ids[0])
                if step > 0 and np.all(np.diff(ids) == step):
                    sl = slice(int(ids[0]), int(ids[-1])+step, step)
        if sl is not None:
            # Native stations commonly request one fixed sample per entity.
            # A regular positive stride is one HDF5 hyperslab, not thousands
            # of individual fancy-index runs. Only requested rows are returned.
            result = dataset[sl] if row is None else dataset[row, sl]
        else:
            unique, inverse = np.unique(ids, return_inverse=True)
            result = (dataset[unique] if row is None else dataset[row, unique])[inverse]
        if result.dtype.kind == "f" and not np.isfinite(result).all():
            raise ValidationError("archive engineering data contain nonfinite values")
        return result
    def _uniform_read(self, dataset, indices):
        samples = int(dataset.attrs["samples_per_entity"])
        ids = np.asarray(indices, dtype=np.int64)
        result = np.empty((len(ids), dataset.shape[2]), dtype=np.float64)
        stations = ids % samples
        for station in np.unique(stations):
            self._check_cancel()
            positions = np.flatnonzero(stations == station)
            entities = ids[positions] // samples
            if len(entities) == 1:
                selector = slice(int(entities[0]), int(entities[0])+1)
                inverse = None
            else:
                step = int(entities[1])-int(entities[0])
                if step > 0 and np.all(np.diff(entities) == step):
                    selector = slice(int(entities[0]), int(entities[-1])+step, step)
                    inverse = None
                else:
                    selector, inverse = np.unique(entities, return_inverse=True)
            block = dataset[selector, int(station), :]
            result[positions] = block if inverse is None else block[inverse]
        if not np.isfinite(result).all():
            raise ValidationError("archive engineering data contain nonfinite values")
        return result

    def _native_stored_read(self,dataset,indices,row):
        samples=int(dataset.attrs["samples_per_entity"])
        # Positive native station stride is a direct single hyperslab with no
        # selector/scatter allocation and no decompression of other stations.
        if isinstance(indices,range) and indices.step>0 and indices.step%samples==0:
            if not len(indices):return np.empty(0,dtype=np.uint64)
            sample=indices.start%samples
            return dataset[row,slice(indices.start//samples,indices[-1]//samples+1,indices.step//samples),sample]
        ids=np.asarray(indices,dtype=np.int64)
        if len(ids)==1:
            return dataset[row,slice(int(ids[0])//samples,int(ids[0])//samples+1),int(ids[0])%samples]
        if len(ids)>1:
            step=int(ids[1])-int(ids[0])
            if step>0 and step%samples==0 and np.all(np.diff(ids)==step):
                return dataset[row,slice(int(ids[0])//samples,int(ids[-1])//samples+1,step//samples),int(ids[0])%samples]
        result=np.empty(len(ids),dtype=np.uint64)
        stations=ids%samples
        for sample in np.unique(stations):
            self._check_cancel()
            positions=np.flatnonzero(stations==sample)
            entities=ids[positions]//samples
            selector,inverse=np.unique(entities,return_inverse=True)
            block=dataset[row,selector,int(sample)]
            result[positions]=block[inverse]
        return result

    def _verify_response_codec(self, key, reserve):
        if key in self._verified_codecs or self._response_layouts.get(key) is None:
            return
        ds = self._component_dataset(key, "responses")
        assoc = self._component_dataset(key, "associations")
        width = int(np.prod(assoc.shape[1:])) if assoc.ndim > 1 else 1
        # Hash and group proof are bounded by remaining working memory. Exact
        # source-table binding persists independently of evictable numeric blocks.
        cost = max(64, width*16)
        self._reserve(reserve+cost)
        step = max(1, min(4096, (self.working_bytes-reserve)//cost))
        digest = hashlib.sha256()
        samples = self._response_layouts[key]
        previous = None
        for start in range(0, len(assoc), step):
            stop = min(start+step, len(assoc))
            self._reserve(reserve+(stop-start)*cost)
            block = self._association_block(key, start, stop)
            digest.update(np.asarray(block, dtype="<i8").tobytes(order="C"))
            leading = block if block.ndim == 1 else block[:,0]
            if len(leading):
                if previous is not None and ((start % samples != 0) != (int(leading[0]) == previous)):
                    raise ValidationError("uniform sample association grouping mismatch")
                boundaries = np.arange(start+1, stop, dtype=np.int64) % samples == 0
                if np.any((leading[1:] != leading[:-1]) != boundaries):
                    raise ValidationError("uniform sample association grouping mismatch")
                previous = int(leading[-1])
        expected = ds.attrs["association_sha256"]
        if isinstance(expected, bytes): expected = expected.decode("utf-8")
        if digest.hexdigest() != expected:
            raise ValidationError("uniform sample association binding mismatch")
        # Verify unique group IDs independently of the digest. Sorting proof
        # buffers are temporary, bounded, and never become a permutation map.
        # With a tiny cap, compare bounded group partitions instead of retaining
        # all IDs. Normal native layouts fit one compact group-ID proof buffer.
        groups = len(assoc)//samples
        proof_cost = max(96, width*16+64)
        self._reserve(reserve+proof_cost)
        capacity = max(1, min(groups, (self.working_bytes-reserve)//proof_cost))
        for first in range(0, groups, capacity):
            stop = min(first+capacity, groups)
            self._reserve(reserve+(stop-first)*proof_cost)
            values = self._spatial_read(assoc, range(first*samples, stop*samples, samples))
            self._association_reads += 1
            ordered = np.sort(values if values.ndim == 1 else values[:,0])
            if np.any(ordered[1:] == ordered[:-1]):
                raise ValidationError("uniform sample group IDs must be unique")
            for other in range(0, first, capacity):
                self._check_cancel()
                end = min(other+capacity, first)
                values = self._spatial_read(assoc, range(other*samples, end*samples, samples))
                self._association_reads += 1
                ids = values if values.ndim == 1 else values[:,0]
                found = np.searchsorted(ordered, ids)
                safe = np.minimum(found, len(ordered)-1)
                if np.any((found < len(ordered)) & (ordered[safe] == ids)):
                    raise ValidationError("uniform sample group IDs must be unique")
        self._check_cancel()
        self._verified_codecs.add(key)

    @_io_boundary
    def _coefficient_row(self, case_index):
        key = ("coefficients", self.model_identity, self.mesh_identity, case_index)
        coefficients = self._cache.get(key)
        if coefficients is not None:
            self._coefficient_hits += 1
            return coefficients
        coefficients = self._handle["coefficients"][case_index]
        self._reads += 1
        self._coefficient_reads += 1
        if not np.isfinite(coefficients).all():
            raise ValidationError("nonfinite coefficients")
        return self._cache_put(key, _freeze(coefficients))
    def _association_block(self, key, start, stop):
        cache_key = ("association_block", self.model_identity, self.mesh_identity, key, start, stop)
        values = self._cache.get(cache_key)
        if values is not None:
            self._association_hits += 1
            return values
        values = _freeze(self._spatial_read(self._component_dataset(key, "associations"), range(start, stop)))
        self._association_reads += 1
        return self._cache_put(cache_key, values)
    def _verify_dependency_associations(self, case_index, key, reserve, seen=None):
        # The reader owns an immutable archive handle. A completed exact
        # whole-table proof is compact metadata tied to this model/mesh and
        # persists when evictable numeric buffers leave the LRU.
        self._verify_response_codec(key, reserve)
        spec = self._meta.get("derived", {}).get(key)
        if spec is None:
            return
        seen = set() if seen is None else seen
        if key in seen or case_index in self._stored_rows[key]:
            return
        seen.add(key)
        for dep in spec["inputs"].values():
            edge = (self.model_identity, self.mesh_identity, key, dep)
            if edge not in self._verified_associations:
                source = self._component_dataset(key, "associations")
                target = self._component_dataset(dep, "associations")
                if source.shape != target.shape:
                    raise ValidationError("derived associations differ")
                width = int(np.prod(source.shape[1:])) if source.ndim > 1 else 1
                sample_cost = max(32, width * 32)
                count = len(source)
                self._reserve(reserve + (sample_cost if count else 0))
                step = max(1, min(CHUNK_BYTES // max(8, width * 8),
                                  (self.working_bytes-reserve) // sample_cost))
                for start in range(0, count, step):
                    stop = min(start+step, count)
                    self._reserve(reserve + (stop-start) * sample_cost)
                    a = self._association_block(key, start, stop)
                    b = self._association_block(dep, start, stop)
                    if not np.array_equal(a, b):
                        raise ValidationError("derived associations differ")
                self._check_cancel()
                self._verified_associations.add(edge)
                self._association_verifications += 1
            self._verify_dependency_associations(case_index, dep, reserve, seen)
    def _tile_size(self, key, reserve, cost):
        # Storage stays at 256 KiB; derived evaluation can amortize dispatch
        # across a 64 MiB managed compute tile, capped by working capacity.
        target = COMPUTE_BYTES if key in self._meta.get("derived", {}) else CHUNK_BYTES
        size = max(1, min(target // max(8, cost), (self.working_bytes-reserve) // cost))
        # Compatible sibling derived operators share a stable basis tile grid.
        # Rounding down preserves both byte ceilings and improves reuse.
        if key in self._meta.get("derived", {}) and size > 1:
            size = 1 << (size.bit_length()-1)
        return size
    def _prepared_runtime(self):
        runtime = self._runtime
        prepare = getattr(runtime, "prepare_response", None)
        reconstruct = getattr(runtime, "reconstruct_prepared_response", None)
        if prepare is None and reconstruct is None:
            return False
        if not callable(prepare) or not callable(reconstruct):
            raise RuntimeDependencyError("lightweight runtime has an incomplete prepared response ABI")
        return True
    def _prepared_response(self, key, indices, active, basis_key, case_index):
        # Preparation preserves the solver-owned original active-column order.
        # Its numeric baseline/selected buffers share the byte-accounted LRU.
        active_tuple = tuple(range(1, len(self._meta["patterns"])+1)) if active is None else tuple(active)
        selection_key = _selection_identity(indices)
        identity = ("prepared_response", self.model_identity, self.mesh_identity, key, selection_key, active_tuple, self._meta.get("combination","legacy_blas/1"))
        prepared = self._cache.get(identity)
        if prepared is not None:
            self._prepared_response_hits += 1
            return prepared
        basis = self._cache.get(basis_key)
        if basis is None:
            basis = _freeze(np.asfortranarray(self._spatial_read(self._component_dataset(key, "responses"), indices)))
            self._cache_put(basis_key, basis)
        prepared = self._runtime_call(case_index,key,self._runtime.prepare_response,basis,active_indices=active,**self._runtime_kwargs())
        expected = len(indices) * (len(active_tuple)+1) * np.dtype("float64").itemsize
        size = getattr(prepared, "nbytes", None)
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or int(size) != expected:
            raise ValidationError("prepared runtime must account exact baseline and selected float64 buffer bytes")
        if self._runtime_kwargs() and getattr(prepared,"combination",None) != self._meta.get("combination","legacy_blas/1"):
            raise RuntimeDependencyError("prepared recovery combination mismatch")
        self._prepared_response_reads += 1
        return self._cache_put(identity, prepared)
    def _field_cost(self, key, case_index):
        # Legacy providers retain their established one-argument cost hook.
        if type(self)._cost is not ResultsReader._cost:
            return self._cost(key)
        return self._cost(key,case_index)

    def _derivation_configuration(self):
        runtime=self._runtime
        capability=runtime is not None and getattr(runtime,"DERIVATION_EXECUTION_API",None)==1
        workers=(2 if self.derivation_workers is None else self.derivation_workers) if capability else 1
        return capability,workers

    def _cost(self, key, case_index=None):
        self._check_cancel()
        if key in getattr(self,"_stored_codecs",()) and (case_index is None or case_index in self._stored_rows[key]):
            # Native selector/scatter matrices, finite masks, predictor and XOR
            # buffers are reserved conservatively before any selected tile.
            return 24*(len(self._meta["patterns"])+1)+192+(80 if self._response_layouts.get(key) is not None else 0)
        # Conservative upper bound for block arrays, fancy index sorting,
        # dependency results, and float64 arithmetic temporaries.
        spec = self._meta["derived"].get(key)
        if spec is not None:
            assoc = self._component_dataset(key, "associations")
            width = int(np.prod(assoc.shape[1:])) if assoc.ndim > 1 else 1
            # Runtime source uses top/bottom vector temporaries for VM and
            # 3x3 tensors, symmetry/isclose/eigen workspaces for principal.
            # Budgets add conservative arrays only; equations stay runtime-owned.
            operator = spec.get("operation")
            workspace = 512 if operator == "shell_principal_stress" else 256 if operator in ("shell_von_mises", "beam_von_mises") else 64 if operator == "equivalent_stress" else 1024
            # Keep the existing principal workspace plus a conservative pool
            # allowance (256KiB at the runtime 4096-row parallel threshold).
            parallel=operator=="shell_principal_stress" and self._derivation_configuration()[1]==2
            return workspace + (64 if parallel else 0) + 24 * width + sum(self._cost(dep,case_index) for dep in spec["inputs"].values())
        matrix_buffers = 24 if self._prepared_runtime() else 16
        return matrix_buffers * (len(self._meta["patterns"])+3) + 64 + (80 if getattr(self, "_response_layouts", {}).get(key) is not None else 0)
    def _prediction_block(self,case_index,key,indices,reserved_bytes):
        row=self._stored_rows[key].get(case_index)
        if row is None:raise UnavailableFieldError("storage predictor cannot recover a missing explicit source row")
        group=self._group(key);step=group["values"].chunks[1];count=self._counts[key]
        self._reserve(reserved_bytes+len(indices)*64+len(self._meta["patterns"])*32)
        output=np.empty(len(indices),dtype=np.float64)
        ids=np.asarray(indices,dtype=np.int64)
        tiles=np.unique(ids//step)
        weights=None
        for tile in tiles:
            self._check_cancel();first=int(tile)*step;stop=min(first+step,count)
            identity=(_prediction.CODEC,self.model_identity,self.mesh_identity,self._prediction_identity,case_index,key,first)
            values=self._cache.get(identity)
            if values is None:
                tile_work=_prediction.work_bytes(self._prediction_info,key,stop-first,len(self._meta["patterns"])+1)
                active_work=reserved_bytes+len(ids)*64+tile_work
                if active_work>_prediction.WORK_BYTES:raise WorkingSetError('storage integrity read exceeds128MiB active workspace')
                self._reserve(active_work)
                if weights is None:weights=self._coefficient_row(case_index)
                active_bases={}
                def basis(primitive,a,b):
                    if primitive not in active_bases:
                        self._check_cancel()
                        basis_key=('storage_prediction_basis/1',self.model_identity,self.mesh_identity,self._prediction_identity,primitive,a,b)
                        table=self._cache.get(basis_key)
                        if table is None:
                            table=_prediction.admit_basis(self._handle['storage_prediction'][self._groups[primitive]]['responses'][a:b],b-a,len(self._meta['patterns'])+1)
                            self._prediction_basis_reads+=1;self._reads+=1
                            table=self._cache_put(basis_key,table)
                        else:self._prediction_basis_hits+=1
                        active_bases[primitive]=table
                    return active_bases[primitive]
                values=_prediction.decode_tile(self._handle,group,self._prediction_info,key,row,case_index,first,stop,weights,self._meta["cases"][case_index].get("active_indices"),self._groups,self._check_cancel,basis=basis)
                active_bases.clear()  # Decoded words do not alias primitive bases.
                self._prediction_tile_reads+=1;self._reads+=1
                values=self._cache_put(identity,values)
            # Output/IDs/unique tiles, predicates, selected IDs and RHS gather
            # arrays coexist; a cached source tile is still an active reference.
            scatter_work=reserved_bytes+len(ids)*64+values.nbytes+len(self._meta["patterns"])*32
            if scatter_work>_prediction.WORK_BYTES:raise WorkingSetError("storage scatter exceeds128MiB active workspace")
            self._reserve(scatter_work)
            selected=np.flatnonzero(ids//step==tile)
            output[selected]=values[ids[selected]-first]
        return output

    def _block(self, case_index, key, indices, memo=None, reserved_bytes=0):
        self._check_cancel()
        memo = {} if memo is None else memo
        if key in memo: return memo[key]
        group = self._group(key)
        if key in self._prediction_codecs:
            result=self._prediction_block(case_index,key,indices,reserved_bytes)
            memo[key]=result
            return result
        row = self._stored_rows[key].get(case_index)
        if row is not None:
            result = self._spatial_read(self._component_dataset(key, "values"), indices, row=row)
            if key in self._stored_codecs:
                selection_key = _selection_identity(indices)
                basis_key = ("stored_predictor_basis",self.model_identity,self.mesh_identity,key,selection_key)
                basis = self._cache.get(basis_key)
                if basis is None:
                    basis = _freeze(np.asfortranarray(self._spatial_read(self._component_dataset(key,"responses"),indices)))
                    self._cache_put(basis_key,basis)
                result = decode_stored(result,basis,self._coefficient_row(case_index),self._meta["cases"][case_index].get("active_indices"))
                self._stored_decodes += 1
        else:
            case = self._meta["cases"][case_index]
            if not case["certified"]: raise UnavailableFieldError("uncertified case has no explicit stored field")
            if self._descriptors[key]["availability"] == "unavailable": raise UnavailableFieldError(f"{key}: unavailable")
            runtime = self._runtime_module()
            spec = self._meta["derived"].get(key)
            if spec is not None:
                inputs = {name: self._block(case_index, dep, indices, memo,reserved_bytes) for name, dep in spec["inputs"].items()}
                # Exact complete association proofs precede tile evaluation.
                capability,workers=self._derivation_configuration()
                execution={}
                kwargs={"workers":workers,"execution":execution} if capability else {}
                result = self._runtime_call(case_index,key,runtime.derive_field,spec,inputs,**kwargs)
                self._derivation_calls+=1
                self._derivation_effective_workers=execution.get("workers",1)
                if self._derivation_effective_workers==2:self._derivation_parallel_calls+=1
            elif key in self._response_keys:
                basis_key = ("basis",key,_selection_identity(indices),self.model_identity,self.mesh_identity)
                coefficients = self._coefficient_row(case_index)
                active = case.get("active_indices")
                if active is not None:
                    # Original solver packets index response columns (baseline=0)
                    # and retain coefficients only in that declared active order.
                    coefficients = coefficients[np.asarray(active, dtype=np.int64)-1]
                if self._prepared_runtime():
                    prepared = self._prepared_response(key, indices, active, basis_key, case_index)
                    result = self._runtime_call(case_index,key,runtime.reconstruct_prepared_response,prepared,coefficients,**self._runtime_kwargs())
                    self._prepared_reconstructions += 1
                else:
                    basis = self._cache.get(basis_key)
                    if basis is None:
                        # F-contiguous cached source columns make the unchanged
                        # runtime active-column selection contiguous. C+F buffers
                        # are reserved before computation, retaining precision.
                        basis = _freeze(np.asfortranarray(self._spatial_read(self._component_dataset(key, "responses"), indices)))
                        self._cache_put(basis_key, basis)
                    result = self._runtime_call(case_index,key,runtime.reconstruct_response,basis,coefficients,active_indices=active,**self._runtime_kwargs())
            else: raise UnavailableFieldError(f"{key}: no stored row for this case")
            self._reconstructions += 1
        result = np.asarray(result, dtype=np.float64)
        if result.shape != (len(indices),) or not np.isfinite(result).all():
            raise ValidationError("runtime returned malformed engineering field")
        memo[key] = result
        return result
    @_io_boundary
    @_cancellable
    def read_field(self, case_id, quantity_id, component, selection=None, *, cancel_event=None):
        with self._lock:
            self._ensure_open()
            case_index, _ = self._case(case_id)
            key = self._key(quantity_id, component)
            self._admit(case_index, key)
            indices, selection_key = self._selection(self._counts[key], selection)
            cache_key = ("field", self.model_identity, self.mesh_identity, case_id, key, selection_key)
            hit = self._cache.get(cache_key)
            if hit is not None: return hit
            output_bytes = len(indices) * 8
            index_bytes = len(indices)*16 if isinstance(indices, np.ndarray) else 0
            cost = self._field_cost(key,case_index)
            self._reserve(output_bytes+index_bytes+(cost if len(indices) else 0)+len(self._meta["patterns"])*32)
            reserve = output_bytes+index_bytes+len(self._meta["patterns"])*32
            self._verify_dependency_associations(case_index, key, reserve)
            chunk = self._tile_size(key, reserve, cost)
            result = np.empty(len(indices), dtype=np.float64)
            for start in range(0, len(indices), chunk):
                stop = min(start+chunk, len(indices))
                self._reserve(output_bytes+index_bytes+(stop-start)*cost+len(self._meta["patterns"])*32)
                self._compute_tiles += 1
                self._largest_compute_tile = max(self._largest_compute_tile, stop-start)
                result[start:stop] = self._block(case_index, key, indices[start:stop],**({"reserved_bytes":reserve} if self._prediction_info is not None else {}))
            return self._cache_put(cache_key, _freeze(result))
    @_io_boundary
    @_cancellable
    def associations(self, quantity_id, component, selection=None, *, cancel_event=None):
        with self._lock:
            self._ensure_open(); key = self._key(quantity_id, component)
            group = self._group(key)
            if "associations" not in group: raise UnavailableFieldError("descriptor has no associations")
            ds = group["associations"]
            indices, selection_key = self._selection(len(ds), selection)
            cache_key = ("association_view", self.model_identity, self.mesh_identity, key, selection_key)
            hit = self._cache.get(cache_key)
            if hit is not None: return hit
            width = int(np.prod(ds.shape[1:])) if ds.ndim > 1 else 1
            self._reserve(len(indices)*(width*16+32))
            values = _freeze(self._spatial_read(ds, indices))
            self._association_reads += 1
            return self._cache_put(cache_key, values)
    @_io_boundary
    @_cancellable
    def geometry(self, selection=None, *, cancel_event=None):
        with self._lock:
            self._ensure_open()
            group = self._handle["geometry"]
            total = sum(ds.size*ds.dtype.itemsize for ds in group.values())
            self._reserve(total)
            self._reads += len(group)
            return {name: _freeze(ds[()]) for name, ds in group.items()}
    @_io_boundary
    @_cancellable
    def read_envelope(self, quantity_id, component, *, case_ids=None, selection=None, strict_failed=False, cancel_event=None):
        with self._lock:
            self._ensure_open(); key = self._key(quantity_id, component)
            chosen = list(self._case_index) if case_ids is None else list(case_ids)
            if len(chosen) != len(set(chosen)): raise ValidationError("envelope case IDs must be unique")
            admitted, failed = [], []
            for cid in chosen:
                try: admitted.append((cid, self._case(cid)[0]))
                except FailedCaseError: failed.append(cid)
            if type(strict_failed) is not bool: raise TypeError("strict_failed must be a boolean")
            if (strict_failed and failed) or not admitted:
                error=FailedCaseError(f"envelope excludes failed cases {tuple(failed)!r}; no completed cases" if not admitted else f"envelope includes failed cases {tuple(failed)!r}")
                error.failed_case_ids=tuple(failed);error.requested_case_ids=tuple(chosen)
                raise error
            coverage=tuple(cid for cid,_ in admitted)
            unavailable=[]
            for cid,ci in admitted:
                try: self._admit(ci,key,runtime_required=False)
                except UnavailableFieldError: unavailable.append((cid,ci))
            if unavailable:
                unavailable.sort(key=lambda item:((index if type(index:=self._meta["cases"][item[1]].get("input_index")) is int and index>=0 else item[1]),item[1]))
                ids=tuple(cid for cid,_ in unavailable)
                error=UnavailableFieldError(f"envelope component {key!r} unavailable for cases {ids!r}",unavailable_case_ids=ids)
                error.requested_case_ids=tuple(chosen);error.failed_case_ids=tuple(failed)
                raise error
            for _,ci in admitted: self._admit(ci,key)
            indices, _ = self._selection(self._counts[key], selection)
            tie_policy = self._descriptors[key].get("tie_policy", "first")
            if tie_policy not in ("first", "last", "exact_earliest_input_index"):
                raise ValidationError("unsupported authoritative envelope tie policy")
            if tie_policy == "exact_earliest_input_index":
                # Only exact equal float64 values tie. The original solver input
                # index remains authoritative even for reversed requested subsets.
                admitted.sort(key=lambda item: self._meta["cases"][item[1]]["input_index"])
            width = max(1, max(len(cid) for cid, _ in admitted))
            outbytes = len(indices)*(16+8*width)
            index_bytes = len(indices)*16 if isinstance(indices, np.ndarray) else 0
            cost = max(self._field_cost(key,ci) for _,ci in admitted)+32+8*width
            reserve = outbytes+index_bytes+len(self._meta["patterns"])*32
            self._reserve(reserve+(cost if len(indices) else 0))
            for _, ci in admitted:
                self._verify_dependency_associations(ci, key, reserve)
            chunk = self._tile_size(key, reserve, cost)
            low, high = np.empty(len(indices)), np.empty(len(indices))
            lowcase, highcase = np.empty(len(indices), dtype=f"U{width}"), np.empty(len(indices), dtype=f"U{width}")
            for start in range(0,len(indices),chunk):
                stop=min(start+chunk,len(indices)); ids=indices[start:stop]
                self._reserve(reserve+(stop-start)*cost)
                self._compute_tiles += 1
                self._largest_compute_tile = max(self._largest_compute_tile, stop-start)
                for j,(cid,ci) in enumerate(admitted):
                    values=self._block(ci,key,ids,**({"reserved_bytes":reserve} if self._prediction_info is not None else {}))
                    if j==0:
                        low[start:stop]=high[start:stop]=values
                        lowcase[start:stop]=highcase[start:stop]=cid
                    else:
                        lt=values<=low[start:stop] if tie_policy=="last" else values<low[start:stop]
                        gt=values>=high[start:stop] if tie_policy=="last" else values>high[start:stop]
                        low[start:stop][lt]=values[lt]; lowcase[start:stop][lt]=cid
                        high[start:stop][gt]=values[gt]; highcase[start:stop][gt]=cid
            self._check_cancel()
            return Envelope(*(_freeze(a) for a in (low,high,lowcase,highcase)), admitted_case_ids=coverage, failed_case_ids=tuple(failed), requested_case_ids=tuple(chosen))

def open_results(path, **kwargs):
    """Open native or supported legacy HDF5 with lifetime owned by the reader."""
    try:
        with h5py.File(path, "r") as handle: schema = _text(handle.attrs.get("schema", ""))
    except OSError as exc:
        raise ValidationError(f"cannot open result archive: {exc}") from exc
    if schema == "anyfem.result":
        from .legacy import LegacyResultsReader
        return LegacyResultsReader(path, **kwargs)
    return ResultsReader(path, **kwargs)
