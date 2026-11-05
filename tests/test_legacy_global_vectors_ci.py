"""Explicit global (frame, component) v1 data; no entity-axis inference."""
import json
from threading import Event
from concurrent.futures import CancelledError
import h5py
import numpy as np
import pytest
from anypost import open_results,ValidationError,UnavailableFieldError,FailedCaseError


def fixture(path,*,location='global',names=('x','y','z'),frames=(0.,2.),dtype='float64',partial=False):
    values=np.array([[0.,-0.,3.],[-4.,5.,-6.]],dtype=dtype)
    with h5py.File(path,'w') as h:
        h.attrs.update(schema='anyfem.result',version=1,complete=True,partial=partial,model_hash='model',mesh_hash='mesh')
        h.create_group('frames').create_dataset('values',data=[0.,1.,2.])
        g=h.create_group('fields').create_group('impulse')
        g.attrs.update(location=location,components=json.dumps(names),unit='N*s',basis='global')
        if frames is not None:g.attrs['frames']=json.dumps(frames)
        g.create_dataset('values',data=values)
    return values


def test_explicit_global_vectors_keep_bits_partial_root_map_and_scalar_sample(tmp_path):
    path=tmp_path/'vector.h5';values=fixture(path)
    with open_results(path) as reader:
        descriptors=reader.components('impulse')
        assert len(descriptors)==3
        assert all(d['component_shape']==[] and d['case_ids']==['0','2'] and d['frame_axis']=='root' for d in descriptors)
        for root,local in (('0',0),('2',1)):
            for column,component in enumerate(('x','y','z')):
                actual=reader.read_field(root,'impulse',component)
                assert actual.shape==(1,)
                np.testing.assert_array_equal(actual.view(np.uint64),values[local,column:column+1].view(np.uint64))
        assert np.signbit(reader.read_field('0','impulse','y')[0])
        with pytest.raises(UnavailableFieldError):reader.read_field('1','impulse','x')
        with pytest.raises(UnavailableFieldError):reader.associations('impulse','x')
        token=Event();token.set()
        with pytest.raises(CancelledError):reader.read_field('0','impulse','x',cancel_event=token)


@pytest.mark.parametrize('location',['node','element','history'])
def test_nonglobal_rank_two_vectors_still_refuse(tmp_path,location):
    path=tmp_path/'nonglobal.h5';fixture(path,location=location)
    with pytest.raises(ValidationError,match='ambiguous legacy component axis'):open_results(path)


def test_global_component_count_mismatch_still_refuses(tmp_path):
    path=tmp_path/'mismatch.h5';fixture(path,names=('x','y'))
    with pytest.raises(ValidationError,match='ambiguous legacy component axis'):open_results(path)


@pytest.mark.parametrize('frames,reason',[(None,'partial field rows require explicit descriptor frames'),((0.,),'descriptor frame table disagrees')])
def test_partial_row_count_needs_exact_declared_frames(tmp_path,frames,reason):
    path=tmp_path/'frames.h5';fixture(path,frames=frames)
    with pytest.raises(ValidationError,match=reason):open_results(path)


@pytest.mark.parametrize('frames',[(9.,10.),(0.,0.)])
def test_unproved_axes_remain_scoped_and_never_alias_root_cases(tmp_path,frames):
    path=tmp_path/'scoped.h5';values=fixture(path,frames=frames)
    with open_results(path) as reader:
        d=reader.components('impulse')[0]
        assert d['frame_axis']=='quantity' and d['case_ids']==['field:impulse:0','field:impulse:1']
        with pytest.raises(UnavailableFieldError):reader.read_field('0','impulse','x')
        np.testing.assert_array_equal(reader.read_stored_field(d['case_ids'][1],'impulse','x'),values[1,:1])


@pytest.mark.parametrize('dtype',['complex128','S8'])
def test_global_malformed_numeric_dtype_still_refuses(tmp_path,dtype):
    path=tmp_path/'dtype.h5';fixture(path,dtype=dtype)
    with pytest.raises(ValidationError,match='numeric and frame-major'):open_results(path)


def test_partial_source_status_is_not_promoted(tmp_path):
    path=tmp_path/'partial.h5';values=fixture(path,partial=True)
    with open_results(path) as reader:
        assert reader.case_metadata('0')['status']=='partial'
        with pytest.raises(FailedCaseError):reader.read_field('0','impulse','x')
        np.testing.assert_array_equal(reader.read_stored_field('0','impulse','y').view(np.uint64),values[0,1:2].view(np.uint64))
