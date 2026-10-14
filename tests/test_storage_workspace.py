"""Workspace discriminator supplements literal predictor source inspection."""
import numpy as np
from anypost.stored_codec import predictor_bits

def test_ordered_prediction_has_only_row_sized_owned_scratch(monkeypatch):
    rows,width=17,101
    basis=np.arange(rows*width,dtype=np.float64).reshape(rows,width)[:,::-1]
    basis.setflags(write=False)
    weights=np.linspace(-.2,.3,width-1);active=list(range(width-1,0,-1))
    expected=predictor_bits(basis,weights,active).copy()
    assert np.shares_memory(np.asarray(basis,dtype=np.float64),basis)
    allocations=[];original_empty=np.empty;original_zeros=np.zeros
    def empty(shape,*args,**kwargs):allocations.append(('empty',shape));return original_empty(shape,*args,**kwargs)
    def zeros(shape,*args,**kwargs):allocations.append(('zeros',shape));return original_zeros(shape,*args,**kwargs)
    monkeypatch.setattr(np,'empty',empty);monkeypatch.setattr(np,'zeros',zeros)
    actual=predictor_bits(basis,weights,active)
    assert np.array_equal(actual,expected)
    assert allocations==[('zeros',rows),('empty',rows)]
    assert actual.shape==(rows,) and np.isnan(actual.view(np.float64)).shape==(rows,)
