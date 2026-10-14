from concurrent.futures import CancelledError
from threading import Event, RLock
from queue import Queue
from time import monotonic
import numpy as np
import pytest
from any3dview import MeshHandle
from anypost.presentation import RetainedResultView, PresentationError
from test_presentation import Reader, Viewer, drain


@pytest.mark.parametrize('failure',['raise','malformed','subscribe','events'])
def test_attach_failure_removes_only_owned_mesh_and_preserves_reader(failure):
    reader,viewer=Reader(),Viewer()
    other=MeshHandle(__import__('any3dview').MeshArrays(**reader.geometry()))
    if failure=='raise':reader.components=lambda: (_ for _ in ()).throw(ValueError('header'))
    if failure=='malformed':reader.components=lambda: ({'component':'x'},)
    if failure=='subscribe':viewer.add_frame_presented_listener=lambda cb: (_ for _ in ()).throw(ValueError('subscribe'))
    if failure=='events':viewer.add_frame_presented_listener=None
    with pytest.raises((ValueError,TypeError)):
        RetainedResultView(reader,viewer,Queue().put,on_presented=lambda _:None)
    assert viewer.handle.removed and not other.removed
    assert not reader.reservations


@pytest.mark.parametrize('cooperative',[True,False])
def test_close_cancels_before_reader_accounting_and_timeout_retry(cooperative):
    reader,viewer,queue=Reader(),Viewer(),Queue()
    entered,release,cancelled=Event(),Event(),Event();lock=RLock()
    reader.supports_cancellation=True
    def read(*args,selection=None,cancel_event=None):
        with lock:
            entered.set()
            if cooperative:
                assert cancel_event.wait(1);cancelled.set()
                raise CancelledError()
            assert release.wait(2)
            return np.array([1.,2.,3.])
    reader.read_field=read
    original=reader.set_external_cache_bytes
    def account(*args):
        assert not entered.is_set() or cooperative and cancelled.is_set() or release.is_set()
        with lock:original(*args)
    reader.set_external_cache_bytes=account
    view=RetainedResultView(reader,viewer,queue.put);view.request_component('a','displacement','ux')
    assert entered.wait(1)
    if not cooperative:
        with pytest.raises(TimeoutError):view.close(timeout=.01)
        assert not viewer.handle.removed and not reader.reservations
        release.set()
    view.close(timeout=1);view.close()
    assert viewer.handle.removed and sum(reader.reservations.values())==0
    while not queue.empty():queue.get_nowait()()
    assert view.last_presented is None


def test_atomic_backend_refusal_preserves_scalars_generations_and_accounting():
    reader,viewer,queue=Reader(),Viewer(),Queue()
    reader.components=lambda: ({'quantity_id':'u','component':'x','location':'node'}, {'quantity_id':'s','component':'x','location':'element'})
    reader.associations=lambda q,c,**kw: np.array([42]) if q=='s' else reader.ids
    reader.read_field=lambda c,q,k,**kw: np.array([4.]) if q=='s' else np.array([1.,2.,3.])
    def support(mesh):
        if mesh.element_scalars is not None:raise NotImplementedError('software line field')
    viewer.add_mesh_arrays=lambda mesh: MeshHandle(mesh,validate_mesh=support)
    view=RetainedResultView(reader,viewer,queue.put)
    try:
        view.request_component('a','u','x');drain(queue)
        before=view.handle.mesh;gens=view.handle.generations;reserved=dict(reader.reservations)
        view.request_component('a','s','x');drain(queue)
        assert isinstance(view.error,NotImplementedError)
        assert view.handle.mesh is before and view.handle.generations==gens and reader.reservations==reserved
    finally:view.close()


def test_atomic_domain_switch_one_scalar_generation_and_stable_geometry():
    reader,viewer,queue=Reader(),Viewer(),Queue()
    reader.components=lambda: ({'quantity_id':'u','component':'x','location':'node'}, {'quantity_id':'s','component':'x','location':'element'})
    reader.associations=lambda q,c,**kw: np.array([42]) if q=='s' else reader.ids
    reader.read_field=lambda c,q,k,**kw: np.array([4.]) if q=='s' else np.array([1.,2.,3.])
    view=RetainedResultView(reader,viewer,queue.put)
    try:
        before=view.handle.mesh
        for q in ('u','s','u'):
            generation=view.handle.generations.scalar
            view.request_component('a',q,'x');drain(queue)
            assert view.error is None and view.handle.generations.scalar==generation+1
            assert view.handle.mesh.positions is before.positions and view.handle.mesh.triangles is before.triangles
            assert sum(reader.reservations.values())==(12 if q=='u' else 4)
    finally:view.close()


def test_custom_old_handle_cross_domain_refuses_without_partial_clear(monkeypatch):
    reader,viewer,queue=Reader(),Viewer(),Queue()
    reader.components=lambda: ({'quantity_id':'u','component':'x','location':'node'}, {'quantity_id':'s','component':'x','location':'element'})
    reader.associations=lambda q,c,**kw: np.array([42]) if q=='s' else reader.ids
    reader.read_field=lambda c,q,k,**kw: np.array([4.]) if q=='s' else np.array([1.,2.,3.])
    view=RetainedResultView(reader,viewer,queue.put);monkeypatch.setattr(MeshHandle,"update_scalar_fields",None)
    try:
        view.request_component('a','u','x');drain(queue);before=view.handle.mesh
        view.request_component('a','s','x');drain(queue)
        assert isinstance(view.error,PresentationError) and view.handle.mesh is before
    finally:view.close()
