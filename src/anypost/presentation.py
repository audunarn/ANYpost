"""Retained ANY3dView presentation of model-bound archive fields."""
from __future__ import annotations

from dataclasses import dataclass
from threading import get_ident
from time import perf_counter_ns
from typing import Any, Callable

import numpy as np

from .async_loader import LatestRequestLoader, LoadCompletion


@dataclass(frozen=True)
class PresentedResult:
    request_id: int
    case_id: str
    quantity_id: str
    component: str
    requested_ns: int
    loaded_ns: int
    applied_ns: int
    presented_ns: int

    @property
    def latency_ms(self) -> float:
        return (self.presented_ns - self.requested_ns) / 1e6


class PresentationError(ValueError):
    """A field cannot be mapped to the retained geometry without ambiguity."""


class RetainedResultView:
    """Attach geometry once, then publish scalar changes on the viewer owner.

    Association tables are authoritative. Multiple samples per entity require
    an explicit selection yielding at most one value per displayed entity.
    Unrepresented entities receive NaN, preserving missing-data semantics.
    The reader remains owned by its caller.
    """

    def __init__(self, reader: Any, viewer: Any,
                 dispatch: Callable[[Callable[[], None]], None], *,
                 on_presented: Callable[[PresentedResult], None] | None = None,
                 geometry: Any = None, descriptors: Any = None):
        from any3dview import MeshArrays

        self.reader, self.viewer = reader, viewer
        self._owner = get_ident()
        self._closed = False
        self._on_presented = on_presented
        self.last_error: BaseException | None = None
        self.last_presented: PresentedResult | None = None
        self._pending: tuple | None = None
        self._cache_owner = f"presentation:{id(self)}"
        self._external_bytes = 0
        self._loader = None
        self._unsubscribe = None
        self._cleanup_complete = False
        if on_presented is not None and not callable(on_presented):
            raise PresentationError("on_presented must be callable")
        names = ("positions", "triangles", "lines", "point_indices", "triangle_to_element",
                 "node_ids", "element_ids", "normals", "active_elements", "line_to_element")
        # UI hosts can prefetch archive snapshots on their I/O executor.
        source = reader.geometry() if geometry is None else geometry
        if "positions" not in source or "triangles" not in source:
            raise PresentationError("archive has no displayable retained geometry")
        mesh = MeshArrays(**{name: source[name] for name in names if name in source})
        self.handle = viewer.add_mesh_arrays(mesh)
        try:
            self._ids = {"node": mesh.node_ids, "element": mesh.element_ids}
            self._maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for location, ids in self._ids.items():
                if ids is not None:
                    order = np.argsort(ids, kind="stable")
                    ordered = ids[order]
                    if len(ordered)>1 and np.any(ordered[1:]==ordered[:-1]):
                        raise PresentationError(f"geometry has duplicate {location} identities")
                    self._maps[location] = ordered, order
            rows=reader.components() if descriptors is None else descriptors
            self._descriptors={}
            for d in rows:
                if any(not isinstance(d.get(name),str) or not d[name] for name in ("quantity_id","component","location")):
                    raise PresentationError("malformed presentation descriptor")
                pair=(d["quantity_id"],d["component"])
                if pair in self._descriptors:raise PresentationError("duplicate presentation descriptor")
                self._descriptors[pair]=d
            self._loader=LatestRequestLoader(dispatch)
            subscribe=getattr(viewer,"add_frame_presented_listener",None)
            self._unsubscribe=subscribe(self._presented) if subscribe is not None else None
            if self._unsubscribe is not None and not callable(self._unsubscribe):
                raise PresentationError("frame subscription must return an unsubscribe callback")
            if on_presented is not None and subscribe is None:
                raise PresentationError("viewer does not expose actual frame-presentation events")
        except BaseException as primary:
            self._closed=True
            for cleanup in ((self._loader.close if self._loader is not None else None),self._unsubscribe,self.handle.remove):
                if cleanup is not None:
                    try:cleanup()
                    except BaseException as error:getattr(primary,"add_note",lambda note:None)(f"presentation attachment cleanup failed: {error}")
            raise

    @property
    def latest_error(self) -> BaseException | None:
        return self.last_error or self._loader.dispatch_error

    @property
    def error(self) -> BaseException | None:
        return self.latest_error

    @property
    def latest_request_id(self) -> int:
        return self._loader.latest_request_id

    def _check(self) -> None:
        if get_ident() != self._owner:
            raise RuntimeError("result presentation must run on the owning UI thread")
        if self._closed:
            raise RuntimeError("result presentation is closed")

    def request_component(self, case_id: str, quantity_id: str, component: str,
                          selection: Any = None, *,
                          selection_loader: Callable[[Any], Any] | None = None) -> int:
        self._check()
        if selection_loader is not None and (not callable(selection_loader) or selection is not None):
            raise PresentationError("provide either selection indices or a callable background selection loader")
        descriptor = self._descriptors.get((quantity_id, component))
        if descriptor is None:
            raise PresentationError(f"unknown component {quantity_id}/{component}")
        location = descriptor["location"]
        if location not in self._maps:
            raise PresentationError(f"no unambiguous retained mapping for location {location!r}")
        self.last_error = None
        self._pending = None

        def load(cancel):
            if cancel.is_set():
                return None
            selected = selection if selection_loader is None else selection_loader(cancel)
            if cancel.is_set():
                return None
            kwargs = {"cancel_event":cancel} if getattr(self.reader,"supports_cancellation",False) is True else {}
            values = self.reader.read_field(case_id, quantity_id, component, selection=selected, **kwargs)
            if cancel.is_set():
                return None
            ids = self.reader.associations(quantity_id, component, selection=selected, **kwargs)
            return case_id, quantity_id, component, location, values, ids

        return self._loader.request(load, self._apply)

    def _apply(self, completion: LoadCompletion) -> None:
        self._check()
        if completion.error is not None:
            self.last_error = completion.error
            return
        if completion.value is None:
            return
        case_id, quantity, component, location, values, ids = completion.value
        try:
            values, ids = np.asarray(values), np.asarray(ids)
            if values.ndim != 1 or ids.ndim != 1 or len(values) != len(ids):
                raise PresentationError("scalar values and association IDs must have matching one-dimensional shapes")
            # Bound identity sorting, selection indices and display/packing
            # temporaries before allocating any of them.
            target_size = len(self._maps[location][0])
            temporary_bytes = values.nbytes + ids.nbytes + len(ids)*32 + target_size*12 + self._external_bytes
            if temporary_bytes > self.reader.working_bytes:
                raise PresentationError("presentation conversion exceeds the reader's temporary working limit")
            ordered, order = self._maps[location]
            if np.array_equal(ids, self._ids[location]):
                # Geometry IDs were already validated unique at attachment.
                # Matching order needs neither sorting nor a scatter buffer.
                display = np.asarray(values, dtype=np.float32)
            else:
                if len(np.unique(ids)) != len(ids):
                    raise PresentationError("multiple samples per entity need an explicit sample selection; averaging is not implicit")
                positions = np.searchsorted(ordered, ids)
                if len(ids) and (np.any(positions >= len(ordered)) or not np.array_equal(ordered[positions], ids)):
                    raise PresentationError("field association contains identities absent from the retained geometry")
                display = np.full(len(ordered), np.nan, dtype=np.float32)
                display[order[positions]] = values
            display.flags.writeable = False
            accounting=getattr(self.reader,"set_external_cache_bytes",None)
            previous_bytes=self._external_bytes
            node=display if location=="node" else None
            element=display if location=="element" else None
            atomic=getattr(self.handle,"update_scalar_fields",None)
            if not callable(atomic) and ((location=="node" and self.handle.mesh.element_scalars is not None) or (location=="element" and self.handle.mesh.node_scalars is not None)):
                raise PresentationError("cross-domain scalar changes require an atomic retained handle")
            accepted_bytes=display.nbytes
            reservation_changed=False
            def reserve(candidate):
                nonlocal accepted_bytes,reservation_changed
                accepted_bytes=sum(value.nbytes for value in (candidate.node_scalars,candidate.element_scalars) if value is not None)
                if accounting is not None:
                    accounting(self._cache_owner,accepted_bytes);reservation_changed=True
            try:
                if callable(atomic):atomic(node_scalars=node,element_scalars=element,before_publish=reserve)
                else:
                    if accounting is not None:
                        accounting(self._cache_owner,accepted_bytes);reservation_changed=True
                    setter=self.handle.update_node_scalars if location=="node" else self.handle.update_element_scalars
                    setter(display)
            except BaseException:
                if reservation_changed:accounting(self._cache_owner,previous_bytes)
                raise
            self._external_bytes=accepted_bytes
            self._pending = (completion, case_id, quantity, component, perf_counter_ns())
        except BaseException as error:
            self.last_error = error

    def _presented(self, event: Any) -> None:
        if self._closed or self._pending is None:
            return
        self._check()
        completion, case_id, quantity, component, applied_ns = self._pending
        if completion.request_id != self.latest_request_id:
            self._pending = None
            return
        presented_ns = int(event.presented_ns)
        if presented_ns < applied_ns:
            return
        self._pending = None
        record = PresentedResult(completion.request_id, case_id, quantity, component,
                                 completion.requested_ns, completion.loaded_ns, applied_ns, presented_ns)
        self.last_presented = record
        if self._on_presented is not None:
            self._on_presented(record)

    @property
    def io_active(self) -> bool:
        """Nonblocking worker state; no archive lock or field data is read."""
        if self._loader is None:return False
        with self._loader._lock:return bool(self._loader._active)

    def close(self, *, timeout: float = 5.0) -> None:
        if get_ident()!=self._owner:
            raise RuntimeError("result presentation must close on the owning UI thread")
        if self._cleanup_complete:return
        self._closed=True;self._pending=None
        # Cancellation and retirement precede reader-lock accounting. On a
        # timeout, caller ownership and all resources stay available for retry.
        self._loader.close(timeout=timeout)
        if self._unsubscribe is not None:
            self._unsubscribe();self._unsubscribe=None
        if not getattr(self.handle,"removed",False):self.handle.remove()
        accounting=getattr(self.reader,"set_external_cache_bytes",None)
        if accounting is not None:accounting(self._cache_owner,0)
        self._external_bytes=0;self._cleanup_complete=True
