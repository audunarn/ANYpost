"""Canonical principal-axis facade refusal and cancellation."""
import threading
import numpy as np
import pytest
from anypost import principal_axes, ValidationError, WorkingSetError, RequestCancelled, RuntimeDependencyError


def test_facade_preserves_canonical_values_and_typed_failures():
    tensor=np.diag([3.,2.,1.]);before=tensor.copy()
    np.testing.assert_array_equal(principal_axes(tensor).values,[3.,2.,1.])
    with pytest.raises(ValidationError):principal_axes(tensor,basis=np.zeros((3,3)))
    with pytest.raises(WorkingSetError):principal_axes(np.tile(tensor,(20,1,1)),working_bytes=4096)
    event=threading.Event();event.set()
    with pytest.raises(RequestCancelled):principal_axes(tensor,cancel_event=event)
    np.testing.assert_array_equal(tensor,before)


def test_missing_runtime_capability_is_typed(monkeypatch):
    import builtins
    actual=builtins.__import__
    def refuse(name,*args,**kwargs):
        if name=='anyfesolver_results.principals':raise ImportError('capability absent')
        return actual(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',refuse)
    with pytest.raises(RuntimeDependencyError) as caught:principal_axes(np.eye(3))
    assert isinstance(caught.value.__cause__,ImportError)


def test_facade_preserves_preconversion_budget_and_basis_refusals(monkeypatch):
    import anyfesolver_results.principals as module
    calls=[]
    class HugeVirtualArray(np.ndarray):
        @property
        def shape(self):calls.append(True);return (3,3)
    tensor=np.eye(3);huge=np.broadcast_to(tensor,(2**22,2**22,3,3)).view(HugeVirtualArray)
    def forbid(*args,**kwargs):raise AssertionError('conversion ran before admission')
    monkeypatch.setattr(module.np,'asarray',forbid)
    with pytest.raises(WorkingSetError) as budget:principal_axes(huge,working_bytes=8192)
    assert isinstance(budget.value.__cause__,MemoryError)
    with pytest.raises(ValidationError) as basis:principal_axes(tensor,basis=huge,working_bytes=8192)
    assert isinstance(basis.value.__cause__,ValueError)
    assert calls==[]


def test_facade_late_cancellation_publishes_no_result_or_input_mutation():
    class AtPublication:
        calls=0
        def is_set(self):
            self.calls+=1
            return self.calls==3
    event=AtPublication();tensor=np.diag([3.,2.,1.]);before=tensor.copy()
    with pytest.raises(RequestCancelled):principal_axes(tensor,cancel_event=event)
    assert event.calls==3
    np.testing.assert_array_equal(tensor,before)
