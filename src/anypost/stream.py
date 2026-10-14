"""Atomic streaming native archive sink; no solver import or field retention."""
from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from functools import wraps
import json
import os
import tempfile
import uuid
import numpy as np
import h5py
from . import storage_prediction as _prediction
from .archive import _dataset, _AssociationPool, _writer_cancellation, CHUNK_BYTES, ResultsReader
from .shared_metadata import CaseMetadataWriter, summary
from .codec import response_dataset, response_shape
from .stored_codec import STORED_CODEC, stored_encoding_option, encode_tile_rows, write_corrections, correction_dataset, logical_stored_shape, stored_shape, NATIVE_STORED_CODEC
from .validation import array, validate_capsule, validate_metadata, json_text
from .errors import ValidationError, ClosedReaderError, RequestCancelled


def _event(method):
    @wraps(method)
    def checked(self, *args, **kwargs):
        with self._lock:
            try:
                self._accept_event()
                self._check_cancel()
                result = method(self, *args, **kwargs)
                self._check_cancel()
                return result
            except BaseException as exc:
                try:
                    self._abort_failure = (type(exc).__name__,type(exc).__module__,str(exc))
                finally:
                    self.abort()
                raise
    return checked


class ArchiveSink:
    """Receive the solver's four callbacks; publish on successful context exit.

    Numeric rows are appended directly to a same-directory temporary HDF5.
    The sink retains compact JSON case records and integer row identifiers,
    never detailed solver dictionaries or complete explicit engineering arrays.
    A changing component layout or any superseded row is refused atomically.
    """
    def __init__(self, path, *, compression="lzf", stored_encoding="auto", cancel_event=None, cancel_check=None):
        self._check_cancel = _writer_cancellation(cancel_event, cancel_check)
        if compression not in ("lzf", None):
            raise ValueError("compression must be 'lzf' or None")
        self.path = Path(path)
        self.compression = compression
        self.stored_encoding = stored_encoding_option(stored_encoding)
        self._lock = RLock()
        self._handle = None
        self._temporary = None
        self._state = "new"
        self._meta = None
        self._prediction_info=None
        self._association_pool = _AssociationPool()
        self._metadata_pool = None
        self._cases = []
        self._case_ids = set()
        self._groups = {}
        self._stored_indices = {}
        self._writes = self._peak_write_bytes = 0
        self._peak_encode_work_bytes = 0
        self._attempt_id = uuid.uuid4().hex
        self._diagnostic_path = None
        self._diagnostic_error = None
        self._abort_failure = None

    @property
    def diagnostics(self):
        return dict(state=self._state, cases=len(self._cases), components=len(self._groups),
                    dataset_block_writes=self._writes, peak_write_block_bytes=self._peak_write_bytes,
                    retained_explicit_value_bytes=0, peak_encode_work_bytes=self._peak_encode_work_bytes,
                    shared_case_tables=0 if self._metadata_pool is None else len(self._metadata_pool.tables),
                    shared_case_references=0 if self._metadata_pool is None else len(self._metadata_pool.references),
                    peak_metadata_input_bytes=0 if self._metadata_pool is None else self._metadata_pool.peak_input_bytes)

    @property
    def diagnostic_path(self):
        return self._diagnostic_path

    @property
    def diagnostic_error(self):
        return self._diagnostic_error

    def _local_failure(self, exc):
        meta = self._meta or {}
        return dict(schema="anypost.archive_failure",version=1,status="cancelled" if type(exc).__name__ in ("KeyboardInterrupt","CancelledError","RequestCancelled","SolveCancelled") else "failed",
                    runtime=meta.get("runtime"),model_identity=meta.get("model_identity"),mesh_identity=meta.get("mesh_identity"),
                    error_type=type(exc).__name__,error_module=type(exc).__module__,message=str(exc),
                    input_case_ids=[case["case_id"] for case in self._cases],input_case_ids_complete=False,
                    diagnostics={"last_callback_error":self._abort_failure})

    def _record_local_failure(self, exc):
        try:
            self.capsule_failed(self._local_failure(exc))
        except Exception as error:
            self._diagnostic_error = str(error)
            note = getattr(exc,"add_note",None)
            if note is not None:note(f"ANYpost diagnostic persistence failed: {error}")

    def capsule_failed(self, record):
        """Optional terminal failure callback; preserve one diagnostic per attempt.

        This callback remains available after abort. It never publishes HDF5 or
        certifies a failed input. JSON diagnostics and completed compact records
        survive context cleanup. Publication uses an exclusive atomic hard link.
        """
        with self._lock:
            if self._diagnostic_path is not None:
                return self._diagnostic_path
            if self._state == "closed":
                raise ValidationError("a successfully published stream cannot fail retroactively")
            self.abort()
            error = None
            try:
                if not isinstance(record,Mapping) or record.get("status") not in ("failed","cancelled"):
                    raise ValidationError("terminal failure needs a failed/cancelled JSON mapping")
                payload = json.loads(json_text(record))
            except ValidationError as exc:
                error = exc
                self._diagnostic_error = str(exc)
                payload = dict(schema="anypost.archive_failure",version=1,status="failed",
                               diagnostics_unavailable={"error_type":type(exc).__name__,"message":str(exc)})
            payload["sink"] = dict(attempt_id=self._attempt_id,archive_path=str(self.path),
                                   completed_case_count=len(self._cases),
                                   completed_cases=[summary(case) for case in self._cases] if self._metadata_pool is not None and self._metadata_pool.references else self._cases,
                                   completed_cases_are_summaries=bool(self._metadata_pool is not None and self._metadata_pool.references),
                                   hdf_published=False,diagnostic_record_complete=error is None)
            contents = json_text(payload).encode("utf-8")
            self.path.parent.mkdir(parents=True,exist_ok=True)
            fd,temporary = tempfile.mkstemp(prefix=f".{self.path.name}.{self._attempt_id}.",suffix=".failure.tmp",dir=self.path.parent)
            destination = self.path.with_name(f"{self.path.name}.{self._attempt_id}.failure.json")
            try:
                with os.fdopen(fd,"wb") as stream:
                    stream.write(contents);stream.flush();os.fsync(stream.fileno())
                # Never overwrite earlier diagnostics, including a UUID collision.
                os.link(temporary,destination)
                self._diagnostic_path = destination
            finally:
                if os.path.exists(temporary):os.unlink(temporary)
            if error is not None:raise error
            return destination

    def _accept_event(self):
        if self._state in ("closed", "aborted"):
            raise ClosedReaderError("archive sink is closed or aborted")
        if self._state == "finalized":
            raise ValidationError("no callback is permitted after capsule_end")

    def __enter__(self):
        with self._lock:
            self._accept_event()
            return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            try:
                self._record_local_failure(exc)
            finally:
                self.abort()
        else:
            self.close()
        return False

    def abort(self):
        with self._lock:
            # Observed stop is terminal even if cleanup is itself interrupted.
            if self._state != "closed":
                self._state = "aborted"
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            if self._temporary is not None and os.path.exists(self._temporary):
                os.unlink(self._temporary)
            self._temporary = None
            if self._state != "closed":
                self._state = "aborted"

    def close(self):
        with self._lock:
            if self._state == "closed":
                return
            if self._state != "finalized":
                error = ValidationError("cannot publish stream without a validated capsule_end")
                self._record_local_failure(error)
                self.abort()
                raise error
            try:
                self._check_cancel()
                self._handle.flush()
                self._handle.close()
                self._handle = None
                with open(self._temporary, "r+b") as stream:
                    os.fsync(stream.fileno())
                self._check_cancel()
                os.replace(self._temporary, self.path)
                self._temporary = None
                self._state = "closed"
            except BaseException as exc:
                try:
                    self._record_local_failure(exc)
                finally:
                    self.abort()
                raise

    def _group(self, key):
        if not isinstance(key, str) or not key:
            raise ValidationError("stream component key must be a nonempty string")
        if key not in self._groups:
            name = str(len(self._groups))
            self._groups[key] = name
            self._handle["components"].create_group(name)
        return self._handle["components"][self._groups[key]]

    def _resizable(self, group, name, width=None, *, integer=False, correction=False):
        dtype = np.dtype("uint64" if correction else "int64" if integer else "float64")
        if width is None:
            shape, chunks, maximum = (0,), (32768,), (None,)
        else:
            shape = (0, width)
            if name == "coefficients":
                chunk_width = max(1, min(width, CHUNK_BYTES // 8))
                chunks = (max(1, min(4096, CHUNK_BYTES // (8 * chunk_width))), chunk_width)
            else:
                chunks = (1, max(1, min(width, encode_tile_rows(len(self._meta["patterns"])+1) if correction else CHUNK_BYTES // 8)))
            maximum = (None, width if width else None)
        return group.create_dataset(name, shape=shape, maxshape=maximum, chunks=chunks,
                                    dtype=dtype, compression=self.compression,
                                    shuffle=True, fletcher32=True)

    def _append(self, ds, values):
        values = np.asarray(values)
        start = ds.shape[0]
        ds.resize(start + values.shape[0], axis=0)
        if not values.size:
            return
        selection = (slice(start, ds.shape[0]),) + tuple(slice(0, n) for n in ds.shape[1:])
        for block in ds.iter_chunks(selection):
            local = (slice(block[0].start-start, block[0].stop-start),) + block[1:]
            value = values[local]
            self._peak_write_bytes = max(self._peak_write_bytes, value.size * ds.dtype.itemsize)
            self._writes += 1
            ds[block] = value

    def _same_dataset(self, ds, incoming, label):
        a = incoming if isinstance(incoming, h5py.Dataset) else np.asarray(incoming)
        if ds.ndim == 3 and ds.attrs.get("codec") == "uniform_samples/1":
            samples = int(ds.attrs["samples_per_entity"])
            if a.shape != (ds.shape[0]*samples, ds.shape[2]):
                raise ValidationError(f"{label}: static layout changed during stream")
            step = ds.chunks[0]
            for station in range(samples):
                for start in range(0, ds.shape[0], step):
                    stop = min(start+step, ds.shape[0])
                    old = ds[start:stop, station, :]
                    new = np.asarray(a[start*samples+station:stop*samples+station:samples],dtype=np.float64)
                    if not np.array_equal(old.view(np.uint64),new.view(np.uint64)):
                        raise ValidationError(f"{label}: static values changed during stream")
            return
        if ds.shape != a.shape:
            raise ValidationError(f"{label}: static layout changed during stream")
        if a.size:
            for block in ds.iter_chunks():
                old = ds[block]
                new = a[block]
                if ds.dtype.kind == "f":
                    old = old.view(np.uint64)
                    new = np.asarray(new,dtype=np.float64).view(np.uint64)
                if not np.array_equal(old,new):
                    raise ValidationError(f"{label}: static values changed during stream")

    def _static_component(self, key, name, values, *, integer=False):
        a = array(values, f"{name}/{key}", integer=integer)
        if name == "responses" and (a.ndim != 2 or a.shape[1] != len(self._meta["patterns"]) + 1):
            raise ValidationError("stream response matrix shape mismatch")
        if name == "associations" and a.ndim not in (1, 2):
            raise ValidationError("stream association shape mismatch")
        group = self._group(key)
        if name in group:
            self._same_dataset(group[name], a, f"{name}/{key}")
        else:
            if name == "responses":
                def recorded(nbytes):
                    self._writes += 1
                    self._peak_write_bytes = max(self._peak_write_bytes, nbytes)
                response_dataset(group, a, group.get("associations"), self.compression, _dataset, recorded)
            else:
                self._association_pool(group, name, a, self.compression, integer=integer)

    @_event
    def capsule_begin(self, header):
        if self._state != "new" or not isinstance(header, Mapping):
            raise ValidationError("capsule_begin requires one fresh sink and a mapping")
        if header.get("schema") != "anysolver.recovery_capsule" or type(header.get("version")) is not int or header.get("version") not in (1,2):
            raise ValidationError("stream requires recovery capsule version 1 or 2")
        # Reuse writer geometry/identity/JSON checks without retaining any basis.
        skeleton = dict(header)
        skeleton.update(cases=(), coefficients=np.empty((0,len(header.get("patterns",())))),
                        descriptors=(), associations={}, responses={}, stored={}, derived={})
        skeleton.setdefault("geometry", {})
        skeleton.pop("storage_prediction",None)
        validated = validate_capsule(skeleton)
        excluded = {"schema", "version", "runtime", "model_identity", "mesh_identity", "solver_identity",
                    "patterns", "geometry", "responses", "associations", "derived", "storage_prediction"}
        self._meta = json.loads(json_text(validated))
        self._meta["source_metadata"] = json.loads(json_text({k:v for k,v in header.items() if k not in excluded}))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, self._temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        os.close(fd)
        self._handle = h5py.File(self._temporary, "w", rdcc_nbytes=0)
        self._handle.attrs.update(schema="anypost.results", version=1, complete=False,
                                  runtime=self._meta["runtime"], model_identity=self._meta["model_identity"],
                                  mesh_identity=self._meta["mesh_identity"])
        self._metadata_pool = CaseMetadataWriter(self._handle,self._meta["model_identity"],self._meta["mesh_identity"],self.compression)
        geometry = self._handle.create_group("geometry")
        for name,value in header.get("geometry",{}).items():
            _dataset(geometry,name,value,self.compression,integer=np.asarray(value).dtype.kind in "iu")
        self._handle.create_group("components")
        self._resizable(self._handle,"coefficients",len(self._meta["patterns"]))
        for key,value in header.get("associations",{}).items():
            self._static_component(key,"associations",value,integer=True)
        for key,value in header.get("responses",{}).items():
            self._static_component(key,"responses",value)
        # Header dependencies are compact metadata and may name final-only fields.
        self._header_derived = json.loads(json_text(header.get("derived",{})))
        role=header.get("storage_prediction")
        if role is not None:
            if validated["capsule_version"]!=2:raise ValidationError("storage prediction requires capsule2")
            def group_name(key):self._group(key);return self._groups[key]
            self._prediction_info=_prediction.install(self._handle,role,self._meta["patterns"],group_name,self._association_pool,self.compression)
            self._meta["storage_prediction"]=self._prediction_info
        self._state = "begun"

    @_event
    def capsule_case(self, record, coefficients):
        if self._state != "begun" or not isinstance(record,Mapping):
            raise ValidationError("capsule_case requires a begun stream and case mapping")
        if record.get("case_id") in self._case_ids:
            raise ValidationError("duplicate stream case ID")
        compact = self._metadata_pool.add_case(record,len(self._cases))
        metadata = dict(self._meta,cases=(compact,),descriptors=(),derived={})
        validate_metadata(metadata, allow_unknown_case_components=True)
        if compact["case_id"] in self._case_ids:
            raise ValidationError("duplicate stream case ID")
        coefficients = array(coefficients,"stream coefficient row",ndim=1)
        if coefficients.shape != (len(self._meta["patterns"]),):
            raise ValidationError("stream coefficient row width mismatch")
        self._append(self._handle["coefficients"],coefficients.reshape(1,-1))
        self._cases.append(compact)
        self._case_ids.add(compact["case_id"])

    @_event
    def capsule_stored(self, key, case_indices, values):
        if self._state != "begun":
            raise ValidationError("capsule_stored requires a begun stream")
        indices = array(case_indices,"stream stored case indices",integer=True,ndim=1)
        values = array(values,"stream stored rows",ndim=2)
        if len(indices) != len(values) or np.any(indices<0) or np.any(indices>=len(self._cases)):
            raise ValidationError("stream stored row indices/shape mismatch")
        group = self._group(key)
        existing = self._stored_indices.setdefault(key,set())
        ids = [int(i) for i in indices]
        if len(set(ids)) != len(ids) or any(i in existing for i in ids):
            raise ValidationError("duplicate explicit case/component row")
        if "responses" in group:
            response_shape(group["responses"],values.shape[1],len(self._meta["patterns"])+1)
        if "values" not in group:
            prediction=self.stored_encoding=="auto" and self._prediction_info is not None and key in self._prediction_info["targets"] and values.shape[1]>0
            correction = self.stored_encoding == "auto" and "responses" in group and values.shape[1]>0
            dataset = (_prediction.create_values(group,0,values.shape[1],self.compression,resizable=True,info=self._prediction_info,key=key,width=len(self._meta["patterns"])+1) if prediction else correction_dataset(group,0,values.shape[1],group["responses"],self.compression,resizable=True) if correction else self._resizable(group,"values",values.shape[1]))
            self._resizable(group,"case_indices",integer=True)
        if logical_stored_shape(group["values"])[1] != values.shape[1]:
            raise ValidationError("changing explicit component layout is not exportable")
        dataset = group["values"]
        if dataset.attrs.get("codec")==_prediction.CODEC:
            start=dataset.shape[0]
            for name in ("values","prediction_modes","decoded_sha256"):group[name].resize(start+len(values),axis=0)
            def written(nbytes):self._writes+=1;self._peak_write_bytes=max(self._peak_write_bytes,nbytes)
            def work(nbytes):self._peak_encode_work_bytes=max(self._peak_encode_work_bytes,nbytes)
            _prediction.write_values(self._handle,group,self._prediction_info,key,values,indices,self._handle["coefficients"],self._cases,self._groups,start_row=start,on_write=written,on_work=work,check=self._check_cancel)
        elif dataset.attrs.get("codec") in (STORED_CODEC,NATIVE_STORED_CODEC):
            start = dataset.shape[0]
            dataset.resize(start+len(values),axis=0)
            def written(nbytes):
                self._writes += 1
                self._peak_write_bytes = max(self._peak_write_bytes,nbytes)
            def work(nbytes):self._peak_encode_work_bytes = max(self._peak_encode_work_bytes,nbytes)
            write_corrections(dataset,values,indices,group["responses"],self._handle["coefficients"],self._cases,start_row=start,on_write=written,on_work=work)
        else:
            self._append(dataset,values)
        self._append(group["case_indices"],indices)
        existing.update(ids)

    @_event
    def capsule_end(self, final):
        if self._state != "begun" or not isinstance(final,Mapping):
            raise ValidationError("capsule_end requires a begun stream and final mapping")
        required = {"descriptors","associations","responses","derived","case_count"}
        if not required <= final.keys() or final["case_count"] != len(self._cases):
            raise ValidationError("stream final metadata or case count mismatch")
        superseded = final.get("stored_superseded_cases",{})
        if not isinstance(superseded,Mapping) or any(len(rows) for rows in superseded.values()):
            raise ValidationError("superseded stored rows indicate changing associations; no earlier case may be discarded")
        descriptors = json.loads(json_text(final["descriptors"]))
        derived = dict(self._header_derived)
        for key,spec in final["derived"].items():
            if key in derived and json.loads(json_text(spec)) != derived[key]:
                raise ValidationError("static derived specification changed during stream")
            derived[key] = json.loads(json_text(spec))
        metadata = dict(self._meta,cases=self._cases,descriptors=descriptors,derived=derived)
        validate_metadata(metadata)
        keys = {d["key"] for d in descriptors}
        if final.get("storage_prediction") is None and self._prediction_info is not None:
            # A staged transport role can be withdrawn only before any encoded
            # target exists. Scientific fields and failed cases stay untouched.
            if any(group.get("values") is not None and group["values"].attrs.get("codec")==_prediction.CODEC
                   for group in self._handle["components"].values()):
                raise ValidationError("cannot withdraw an encoded storage prediction role")
            if any(self._stored_indices.get(key) for key in self._prediction_info["targets"]):
                raise ValidationError("cannot withdraw a used storage prediction role")
            for key in set(self._groups)-keys:
                group=self._group(key)
                if len(group):raise ValidationError("cannot withdraw nonempty scientific component")
                del self._handle["components"][self._groups.pop(key)]
            del self._handle["storage_prediction"]
            self._prediction_info=None
            self._meta.pop("storage_prediction",None)
            metadata.pop("storage_prediction",None)
        if set(self._groups)-keys or set(final["associations"])-keys or set(final["responses"])-keys:
            raise ValidationError("stream final descriptor/component identity mismatch")
        for key in keys:
            self._group(key)
        for key,value in final["associations"].items():
            self._static_component(key,"associations",value,integer=True)
        for key,value in final["responses"].items():
            self._static_component(key,"responses",value)
        counts = {}
        for d in descriptors:
            key=d["key"];group=self._group(key);assoc=group.get("associations")
            count=0 if assoc is None else len(assoc);counts[key]=count
            availability=d["availability"]
            if assoc is None and availability!="unavailable":
                raise ValidationError("stream readable field lacks associations")
            if "responses" in group:
                response_shape(group["responses"], count, len(self._meta["patterns"])+1)
            if "values" in group:
                if group["values"].attrs.get("codec")==_prediction.CODEC:_prediction.validate_values(group,len(self._stored_indices[key]),count)
                else:stored_shape(group["values"],(len(self._stored_indices[key]),count),"responses" in group,response=group.get("responses"))
            if availability=="unavailable" and ("values" in group or "responses" in group or key in derived):
                raise ValidationError("unavailable stream field contains success data")
            if availability=="stored" and "values" not in group:
                raise ValidationError("stored stream field has no explicit rows")
            if availability=="reconstructable" and "responses" not in group and key not in derived:
                raise ValidationError("stream reconstructable field has no recovery path")
            if key in derived and availability!="reconstructable":
                raise ValidationError("stream derived field must be reconstructable")
        for key,spec in derived.items():
            for dep in spec["inputs"].values():
                if counts[key]!=counts[dep]:
                    raise ValidationError("stream derived association count mismatch")
                self._same_dataset(self._group(key)["associations"],self._group(dep)["associations"],"derived association")
        role=final.get("storage_prediction")
        if (role is None)!=(self._prediction_info is None):raise ValidationError("storage prediction role changed during stream")
        if role is not None:
            incoming=_prediction.manifest(role,self._meta["patterns"],final["associations"],descriptors)
            if incoming!=self._prediction_info:raise ValidationError("storage prediction metadata changed during stream")
            for key,values in role["responses"].items():
                self._same_dataset(self._handle["storage_prediction"][self._groups[key]]["responses"],values,"storage prediction")
            _prediction.validate_disk(self._handle,self._prediction_info,self._meta["patterns"],self._groups,{d["key"]:d for d in descriptors},counts)
        metadata["component_groups"]=dict(self._groups)
        metadata["stored_case_indices"]={key:self._group(key)["case_indices"][()].tolist() for key in self._stored_indices}
        excluded={"descriptors","associations","responses","derived","case_count","storage_prediction"}
        metadata["source_metadata"].update(json.loads(json_text({k:v for k,v in final.items() if k not in excluded})))
        self._handle.attrs["version"] = self._metadata_pool.finish(metadata)
        self._handle.create_dataset("metadata_json",data=json_text(metadata),dtype=h5py.string_dtype("utf-8"))
        self._check_cancel()
        self._handle.attrs["peak_encode_work_bytes"]=self._peak_encode_work_bytes
        self._handle.attrs["complete"]=True
        self._handle.flush()
        # Reader metadata validation catches layout compatibility without fields.
        with ResultsReader(self._temporary,cache_bytes=0):
            pass
        self._state="finalized"
