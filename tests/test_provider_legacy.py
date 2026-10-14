from threading import Event
import h5py
import numpy as np
import pytest
from anypost import open_results,UnavailableFieldError,WorkingSetError,ValidationError
from concurrent.futures import CancelledError
from test_legacy import legacy_fixture

def test_public_legacy_history_exact_selectors_and_resource_refusal(tmp_path):
    path=tmp_path/'legacy.h5';legacy_fixture(path)
    values=np.arange(12.,dtype=np.float64).reshape(2,2,3)
    with h5py.File(path,'r+') as h:h.create_group('histories').create_dataset('displacement',data=values)
    with open_results(path) as r:
        np.testing.assert_array_equal(r.read_history('displacement',[1,0,1]),values[[1,0,1]])
        np.testing.assert_array_equal(r.read_history('displacement',slice(None,None,-1)),values[::-1])
        with pytest.raises(ValueError):r.read_history('displacement').setflags(write=True)
        with pytest.raises(UnavailableFieldError):r.read_history('absent')
        with pytest.raises(ValidationError):r.read_history('nested/name')
        token=Event();token.set()
        with pytest.raises(CancelledError):r.read_history('displacement',cancel_event=token)
    with open_results(path,working_bytes=100) as r:
        with pytest.raises(WorkingSetError):r.read_history('displacement')


def test_field_specific_legacy_committed_frames_are_not_global_row_indices(tmp_path):
    import json
    path=tmp_path/'different.h5';legacy_fixture(path)
    with h5py.File(path,'r+') as h:
        group=h['fields/stress'];del group['values']
        group.create_dataset('values',data=np.array([[[9.,10.],[11.,12.],[13.,14.]]]))
        group.attrs['frames']=json.dumps([1.])
    with open_results(path) as r:
        assert r.components()[0]['case_ids']==['1']
        np.testing.assert_array_equal(r.read_field('1','stress','x'),[9.,11.,13.])
        with pytest.raises(UnavailableFieldError):r.read_field('0','stress','x')


def test_global_history_field_uses_explicit_frame_axis(tmp_path):
    import json
    path=tmp_path/'global.h5';legacy_fixture(path)
    with h5py.File(path,'r+') as h:
        g=h['fields'].create_group('support_Fx');g.create_dataset('values',data=[100.,200.])
        g.attrs.update(location='global',components=json.dumps(['scalar']),frames=json.dumps([0.,1.]),unit='N')
    with open_results(path) as r:
        np.testing.assert_array_equal(r.read_field('1','support_Fx','scalar'),[200.])


def test_explicit_partial_stored_read_does_not_change_source_status(tmp_path):
    from anypost import FailedCaseError
    path=tmp_path/'partial.h5';values=legacy_fixture(path,partial=True)
    with open_results(path) as r:
        assert r.cases()[1]['status']=='partial'
        np.testing.assert_array_equal(r.read_stored_field('1','stress','x'),values[1,:,0])
        with pytest.raises(FailedCaseError):r.read_field('1','stress','x')
        assert r.cases()[1]['status']=='partial'


def test_legacy_shape_history_table_discovery_never_reads_fields(tmp_path,monkeypatch):
    path=tmp_path/'headers.h5';legacy_fixture(path)
    with h5py.File(path,'r+') as h:h.create_group('histories').create_dataset('u',data=np.arange(6.).reshape(2,3))
    original=h5py.Dataset.__getitem__;field_reads=[]
    def guarded(ds,key):
        if ds.name.startswith('/fields/') and ds.name.endswith('/values'):
            field_reads.append(ds.name);raise AssertionError('discovery must not load fields')
        return original(ds,key)
    monkeypatch.setattr(h5py.Dataset,'__getitem__',guarded)
    with open_results(path) as r:
        descriptor=r.components()[0];metadata=r.metadata()
        assert descriptor['stored_shape']==[2,3,2]
        assert metadata['history_keys']==['u'] and metadata['history_shapes']['u']==[2,3]
        assert metadata['table_shapes']['stress_element_ids']==[3]
        assert metadata['table_dtypes']['stress_element_ids']=='int64'
    assert not field_reads


def _independent_axis(path, frames, values):
    import json
    with h5py.File(path,'r+') as h:
        g=h['fields'].create_group('iterations')
        g.create_dataset('values',data=np.asarray(values,dtype=np.float64).reshape(len(frames),1))
        g.attrs.update(location='history',components=json.dumps(['scalar']),frames=json.dumps(frames),unit='')


def test_independent_quantity_axes_preserve_root_cases_and_local_rows(tmp_path):
    path=tmp_path/'independent.h5';legacy_fixture(path)
    _independent_axis(path,[10.,20.,30.],[100.,200.,300.])
    with open_results(path) as r:
        d=next(d for d in r.components() if d['quantity_id']=='iterations')
        assert d['frames']==[10.,20.,30.] and d['stored_shape']==[3,1]
        assert d['frame_axis']=='quantity' and d['local_frame_indices']==[0,1,2]
        assert d['case_ids']==['field:iterations:0','field:iterations:1','field:iterations:2']
        assert [c['case_id'] for c in r.cases()]==['0','1']
        np.testing.assert_array_equal(r.frames(),[0.,1.])
        assert r.diagnostics['reads']==0
        np.testing.assert_array_equal(r.read_field(d['case_ids'][2],'iterations','scalar'),[300.])
        assert r.case_metadata(d['case_ids'][2])['frame']==30.
        with pytest.raises(UnavailableFieldError):r.read_field(d['case_ids'][0],'stress','x')
        with pytest.raises(UnavailableFieldError):r.read_field('0','iterations','scalar')
        with pytest.raises(IndexError):r.read_stored_field(d['case_ids'][0],'iterations','scalar',[1])
        token=Event();token.set()
        with pytest.raises(CancelledError):r.read_stored_field(d['case_ids'][0],'iterations','scalar',cancel_event=token)


def test_scoped_partial_snapshots_keep_status_and_refuse_normal_access(tmp_path):
    from anypost import FailedCaseError,UnknownCaseError
    path=tmp_path/'independent_partial.h5';legacy_fixture(path,partial=True)
    _independent_axis(path,[1.,2.],[100.,200.])
    with open_results(path) as r:
        d=next(d for d in r.components() if d['quantity_id']=='iterations')
        cid=d['case_ids'][1]
        assert r.case_metadata(cid)['status']=='partial'
        with pytest.raises(FailedCaseError):r.read_field(cid,'iterations','scalar')
        np.testing.assert_array_equal(r.read_stored_field(cid,'iterations','scalar'),[200.])
        with pytest.raises(UnavailableFieldError):r.read_stored_field(cid,'stress','x')
        with pytest.raises(UnknownCaseError):r.read_stored_field('field:iterations:99','iterations','scalar')
        assert r.case_metadata(cid)['status']=='partial' and len(r.cases())==2


def test_ambiguous_duplicate_root_values_are_quantity_scoped(tmp_path):
    import json
    path=tmp_path/'duplicate.h5';legacy_fixture(path)
    with h5py.File(path,'r+') as h:
        h['frames/values'][:]=[1.,1.]
        h['fields/stress'].attrs['frames']=json.dumps([1.,1.])
    _independent_axis(path,[1.],[90.])
    with open_results(path) as r:
        d=next(d for d in r.components() if d['quantity_id']=='iterations')
        assert d['frame_axis']=='quantity' and d['case_ids']==['field:iterations:0']
        np.testing.assert_array_equal(r.read_field(d['case_ids'][0],'iterations','scalar'),[90.])
        with pytest.raises(UnavailableFieldError):r.read_field('0','iterations','scalar')



def test_actual_group_history_contract_headers_selectors_and_no_numeric_discovery(tmp_path,monkeypatch):
    path=tmp_path/'xy.h5';legacy_fixture(path)
    x=np.arange(6.,dtype=np.float64).reshape(3,2);y=x*-2.
    with h5py.File(path,'r+') as h:
        g=h.create_group('histories').create_group('force');g.create_dataset('x',data=x);g.create_dataset('y',data=y)
    original=h5py.Dataset.__getitem__;loads=[]
    def guard(ds,key):
        if ds.name.startswith('/histories/') or ds.name.startswith('/fields/'):
            loads.append(ds.name);raise AssertionError('header discovery must not read values')
        return original(ds,key)
    with monkeypatch.context() as m:
        m.setattr(h5py.Dataset,'__getitem__',guard)
        with open_results(path) as r:assert r.metadata()['history_shapes']['force']==[3,2]
    assert loads==[]
    with open_results(path) as r:
        pair=r.read_history('force',rows=[2,0,2])
        assert isinstance(pair,tuple) and len(pair)==2
        for actual,expected in zip(pair,(x[[2,0,2]],y[[2,0,2]])):
            np.testing.assert_array_equal(actual,expected)
            with pytest.raises(ValueError):actual.setflags(write=True)
        pair[0].shape=(6,)
        np.testing.assert_array_equal(r.read_history('force')[0],x)
        token=Event();token.set()
        with pytest.raises(CancelledError):r.read_history('force',cancel_event=token)
    with open_results(path,working_bytes=100) as r:
        with pytest.raises(WorkingSetError):r.read_history('force')


def test_history_mismatched_group_is_typed_and_other_fields_remain_readable(tmp_path):
    path=tmp_path/'badxy.h5';values=legacy_fixture(path)
    with h5py.File(path,'r+') as h:
        g=h.create_group('histories').create_group('bad');g.create_dataset('x',data=[1.,2.]);g.create_dataset('y',data=[1.])
    with open_results(path) as r:
        with pytest.raises(ValidationError):r.read_history('bad')
        np.testing.assert_array_equal(r.read_field('1','stress','x'),values[1,:,0])


def test_undeclared_1d_spatial_field_does_not_refuse_other_frames(tmp_path):
    path=tmp_path/'unframed.h5';values=legacy_fixture(path)
    with h5py.File(path,'r+') as h:
        g=h['fields'].create_group('raw');g.create_dataset('values',data=[10.,20.,30.]);g.attrs.update(location='global')
    with open_results(path) as r:
        d=next(d for d in r.components() if d['quantity_id']=='raw')
        assert d['stored_shape']==[3] and d['component_shape']==[3]
        assert d['frame_axis']=='quantity' and d['case_ids']==['field:raw:0']
        assert r.case_metadata(d['case_ids'][0])['frame'] is None
        assert len(r.cases())==2
        np.testing.assert_array_equal(r.read_field('1','stress','x'),values[1,:,0])
        np.testing.assert_array_equal(r.read_stored_field(d['case_ids'][0],'raw','scalar',[2,0,2]),[30.,10.,30.])
        with pytest.raises(UnavailableFieldError):r.read_stored_field('1','raw','scalar')
        assert next(d for d in r.metadata()['original_descriptors'] if d['source_key']=='raw').get('key') is None
