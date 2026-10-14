from queue import Queue
from types import SimpleNamespace
from time import perf_counter_ns, monotonic, sleep
import numpy as np
import pytest
from threading import get_ident, Event
from any3dview.retained import MeshHandle
from anypost.presentation import RetainedResultView, PresentationError


class Reader:
    working_bytes = 256*1024**2
    def __init__(self):
        self.ids = np.array([3, 7, 9])
        self.reservations = {}
    def geometry(self):
        return dict(positions=np.array([[0.,0,0],[1.,0,0],[0.,1,0]]),
                    triangles=np.array([[0,1,2]]),node_ids=np.array([7,3,9]),element_ids=np.array([42]))
    def components(self):
        return [dict(quantity_id="displacement",component="ux",location="node")]
    def read_field(self,*args,selection=None):
        data=np.array([10.,20.,30.])
        return data if selection is None else data[selection]
    def associations(self,*args,selection=None):
        return self.ids if selection is None else self.ids[selection]
    def set_external_cache_bytes(self,key,size):
        self.reservations[key]=size


class Viewer:
    def __init__(self): self.listeners=[]
    def add_mesh_arrays(self,mesh): self.handle=MeshHandle(mesh); return self.handle
    def add_frame_presented_listener(self,callback):
        self.listeners.append(callback)
        return lambda: self.listeners.remove(callback)
    def present(self):
        for callback in tuple(self.listeners): callback(SimpleNamespace(presented_ns=perf_counter_ns()))


def drain(queue):
    deadline=monotonic()+2
    while queue.empty():
        assert monotonic()<deadline
        sleep(.002)
    while not queue.empty(): queue.get_nowait()()


def test_identity_mapping_and_presented_boundary_retain_topology():
    reader,viewer,queue=Reader(),Viewer(),Queue()
    presented=[]
    view=RetainedResultView(reader,viewer,queue.put,on_presented=presented.append)
    original=view.handle.mesh.triangles
    view.request_component("case","displacement","ux")
    drain(queue)
    np.testing.assert_array_equal(view.handle.mesh.node_scalars,[20,10,30])
    assert view.handle.generations.topology==0
    assert view.handle.mesh.triangles is original
    assert not presented
    viewer.present()
    assert presented[0].latency_ms>=0
    assert presented[0].presented_ns>=presented[0].applied_ns
    assert sum(reader.reservations.values())==12
    view.close()
    assert view.handle.removed
    assert sum(reader.reservations.values())==0


def test_duplicate_samples_refused_without_mutating_scalars():
    reader,viewer,queue=Reader(),Viewer(),Queue()
    reader.ids=np.array([3,3,9])
    view=RetainedResultView(reader,viewer,queue.put)
    view.request_component("case","displacement","ux")
    drain(queue)
    assert isinstance(view.error,PresentationError)
    assert view.handle.mesh.node_scalars is None
    assert view.handle.generations.scalar==0
    view.close()


def test_subset_missing_entities_are_nan_and_unknown_ids_refuse():
    reader,viewer,queue=Reader(),Viewer(),Queue()
    view=RetainedResultView(reader,viewer,queue.put)
    view.request_component("case","displacement","ux",selection=[0])
    drain(queue)
    assert np.isnan(view.handle.mesh.node_scalars[[0,2]]).all()
    assert view.handle.mesh.node_scalars[1]==10
    before=view.handle.generations.scalar
    reader.ids=np.array([100,7,9])
    view.request_component("case","displacement","ux")
    drain(queue)
    assert isinstance(view.error,PresentationError)
    assert view.handle.generations.scalar==before
    view.close()


def test_background_sample_selection_precedes_field_read_and_preserves_missing():
    reader, viewer, queue = Reader(), Viewer(), Queue()
    reader.ids = np.array([3, 3, 9])
    owner = get_ident()
    calls = []
    def select(cancel):
        assert isinstance(cancel, Event)
        assert get_ident() != owner
        calls.append("native selection")
        return [1, 2]
    view = RetainedResultView(reader, viewer, queue.put)
    view.request_component("case", "displacement", "ux", selection_loader=select)
    drain(queue)
    assert calls == ["native selection"] and view.error is None
    assert np.isnan(view.handle.mesh.node_scalars[0])
    np.testing.assert_array_equal(view.handle.mesh.node_scalars[1:], [20., 30.])
    view.close()


def test_background_selection_failure_and_invalid_arguments_do_not_mutate():
    reader, viewer, queue = Reader(), Viewer(), Queue()
    view = RetainedResultView(reader, viewer, queue.put)
    error = ValueError("unavailable native sample")
    def refuse(cancel):
        raise error
    view.request_component("case", "displacement", "ux", selection_loader=refuse)
    drain(queue)
    assert view.error is error and view.handle.generations.scalar == 0
    with pytest.raises(PresentationError):
        view.request_component("case", "displacement", "ux", [0], selection_loader=refuse)
    with pytest.raises(PresentationError):
        view.request_component("case", "displacement", "ux", selection_loader=123)
    view.close()


def test_superseded_background_selection_does_not_read_or_publish_old_field():
    reader, viewer, queue = Reader(), Viewer(), Queue()
    entered, release = Event(), Event()
    read_cases = []
    ordinary_read = reader.read_field
    def read(case, *args, **kwargs):
        read_cases.append(case)
        return ordinary_read(case, *args, **kwargs)
    reader.read_field = read
    def select(cancel):
        entered.set()
        assert release.wait(2)
        assert cancel.is_set()
        return [0]
    view = RetainedResultView(reader, viewer, queue.put)
    view.request_component("old", "displacement", "ux", selection_loader=select)
    assert entered.wait(2)
    view.request_component("latest", "displacement", "ux", selection=[2])
    release.set()
    drain(queue)
    assert read_cases == ["latest"] and view.error is None
    assert np.isnan(view.handle.mesh.node_scalars[:2]).all()
    assert view.handle.mesh.node_scalars[2] == 30
    view.close()


def test_prefetched_startup_attaches_without_owner_thread_archive_access():
    reader, viewer, queue = Reader(), Viewer(), Queue()
    geometry, descriptors = reader.geometry(), reader.components()
    def refuse():
        raise AssertionError("owner thread accessed the archive at attachment")
    reader.geometry = reader.components = refuse
    view = RetainedResultView(reader, viewer, queue.put,
                              geometry=geometry, descriptors=descriptors)
    np.testing.assert_array_equal(view.handle.mesh.node_ids, [7, 3, 9])
    view.request_component("case", "displacement", "ux")
    drain(queue)
    np.testing.assert_array_equal(view.handle.mesh.node_scalars, [20., 10., 30.])
    view.close()
