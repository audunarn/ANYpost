"""Lossless byte-codec controls, not physical/source qualification fixtures."""
from copy import deepcopy
from concurrent.futures import CancelledError
from threading import Event
import importlib
import h5py
import numpy as np
import pytest
from anypost import write_results,open_results,ArchiveSink
from anypost.errors import ValidationError,WorkingSetError,UnavailableFieldError,RuntimeDependencyError
from anypost import storage_prediction as prediction

def capsule(count=9,derived=False):
    names=('axial_stress','bending_stress_y','bending_stress_z','shear_stress_y','shear_stress_z','torsional_stress') if derived else ('axial_stress',)
    ids=np.arange(101,101+count,dtype=np.int64);x=np.arange(count,dtype=np.float64)
    responses={name:np.column_stack((x*.01+index+1,x*.2-index,np.sin(x*.001+index))) for index,name in enumerate(names)}
    cases=[dict(case_id=f'c{i}',status='converged',certified=True,portable=False,displacement_portable=None,route='codec-test-stored',active_indices=[2,1]) for i in range(2)]
    weights=np.array([[1.25,-.3],[-2.7,.85]])
    specs={}
    if derived:
        specs={'von_mises':{'operation':'beam_von_mises','inputs':{name:name for name in names}},'equivalent_stress':{'operation':'equivalent_stress','inputs':{'von_mises':'von_mises'}}}
    keys=(*names,*specs)
    role=dict(version=1,runtime='anysolver-results/2',combination='ordered_float64/1',responses=responses,derived=specs,associations={key:ids for key in keys},targets=list(keys),provenance={'fixture':'byte-codec-only'})
    info=prediction.manifest(role,('p0','p1'))
    values={key:np.vstack([prediction.predict(info,key,0,count,weights[i],cases[i]['active_indices'],lambda k,a,b:responses[k][a:b]) for i in range(2)]) for key in keys}
    for data in values.values():data[:,0]=-0.;data[:,1]=np.nextafter(data[:,1],np.inf)
    descriptors=[dict(key=key,quantity_id='stress',component=key,location='element',unit='Pa',basis='local',sampling='native_samples',availability='stored') for key in keys]
    return dict(schema='anysolver.recovery_capsule',version=2,runtime='anysolver-results/2',combination='ordered_float64/1',model_identity='codec-fixture',mesh_identity='codec-ids',solver_identity={'kind':'byte-test-not-source-qualification'},geometry={},patterns=('p0','p1'),coefficients=weights,cases=cases,descriptors=descriptors,associations={key:ids for key in keys},responses={},derived={},stored={key:dict(case_indices=np.array([0,1]),values=data) for key,data in values.items()},storage_prediction=role)

def assert_bits(actual,expected):assert np.array_equal(np.asarray(actual).view(np.uint64),np.asarray(expected).view(np.uint64))

@pytest.mark.parametrize('stream',[False,True])
@pytest.mark.parametrize('derived',[False,True])
def test_complete_stored_rows_roundtrip_selectors_and_envelope(tmp_path,stream,derived):
    c=capsule(9,derived);path=tmp_path/'a.h5'
    if stream:
        header={k:v for k,v in c.items() if k not in ('cases','coefficients','stored','descriptors')}
        with ArchiveSink(path) as sink:
            sink.capsule_begin(header)
            for record,weights in zip(c['cases'],c['coefficients']):sink.capsule_case(record,weights)
            for key,row in c['stored'].items():sink.capsule_stored(key,row['case_indices'],row['values'])
            sink.capsule_end(dict(descriptors=c['descriptors'],associations=c['associations'],responses={},derived={},case_count=2,storage_prediction=c['storage_prediction']))
    else:write_results(path,c)
    with open_results(path) as reader:
        assert all(d['availability']=='stored' for d in reader.components())
        assert not reader._response_keys and not reader._meta['derived']
        for key,row in c['stored'].items():
            for index in range(2):
                assert_bits(reader.read_field(f'c{index}','stress',key),row['values'][index])
                assert_bits(reader.read_field(f'c{index}','stress',key,selection=[7,0,7,1]),row['values'][index,[7,0,7,1]])
            envelope=reader.read_envelope('stress',key)
            assert_bits(envelope.minimum,np.min(row['values'],axis=0))
            assert_bits(envelope.maximum,np.max(row['values'],axis=0))

def test_missing_stored_case_never_falls_through_prediction(tmp_path):
    c=capsule();c['stored']['axial_stress']['case_indices']=np.array([0]);c['stored']['axial_stress']['values']=c['stored']['axial_stress']['values'][:1]
    path=write_results(tmp_path/'a.h5',c)
    with open_results(path) as reader:
        with pytest.raises(UnavailableFieldError):reader.read_field('c1','stress','axial_stress')
        assert reader.case_components('c1')[0]['availability']=='unavailable'

@pytest.mark.parametrize('corrupt',['unselected_word','basis','mode','digest'])
def test_partial_selection_verifies_whole_original_tile(tmp_path,corrupt):
    c=capsule(32769);path=write_results(tmp_path/'a.h5',c)
    with h5py.File(path,'r+') as file:
        if corrupt=='unselected_word':file['components/0/values'][0,1000]=int(file['components/0/values'][0,1000])^1
        elif corrupt=='basis':file['storage_prediction/0/responses'][1000,0]+=2.
        elif corrupt=='mode':file['components/0/prediction_modes'][0,0]=9
        else:file['components/0/decoded_sha256'][0,0]=b'0'*64
    with open_results(path) as reader:
        with pytest.raises(ValidationError):reader.read_field('c0','stress','axial_stress',selection=[0])

def test_low_workspace_refuses_full_integrity_tile_before_selected_return(tmp_path):
    path=write_results(tmp_path/'a.h5',capsule(32769))
    with open_results(path,working_bytes=2*1024**2) as reader:
        with pytest.raises(WorkingSetError):reader.read_field('c0','stress','axial_stress',selection=[0])

def test_verified_tile_is_byte_accounted_and_reused(tmp_path):
    c=capsule(32769);path=write_results(tmp_path/'a.h5',c)
    with open_results(path,cache_bytes=1024**2) as reader:
        assert_bits(reader.read_field('c0','stress','axial_stress',selection=[0]),c['stored']['axial_stress']['values'][0,[0]])
        count=reader.diagnostics['storage_prediction_tile_decodes']
        assert_bits(reader.read_field('c0','stress','axial_stress',selection=[1]),c['stored']['axial_stress']['values'][0,[1]])
        assert reader.diagnostics['storage_prediction_tile_decodes']==count
        assert reader.diagnostics['cache_resident_bytes']<=1024**2
    with open_results(path,cache_bytes=0) as reader:
        reader.read_field('c0','stress','axial_stress',selection=[0]);reader.read_field('c0','stress','axial_stress',selection=[1])
        assert reader.diagnostics['cache_resident_bytes']==0
        assert reader.diagnostics['storage_prediction_tile_decodes']==2

def test_metadata_without_runtime_and_new_capability_refusal(tmp_path,monkeypatch):
    import anysolver_results as runtime
    path=write_results(tmp_path/'a.h5',capsule())
    monkeypatch.setattr(runtime,'STORAGE_PREDICTION_API',0)
    with open_results(path) as reader:
        assert reader.components()[0]['availability']=='stored'
        with pytest.raises(RuntimeDependencyError):reader.read_field('c0','stress','axial_stress')

def test_pre_cancelled_selection_exposes_no_values(tmp_path):
    path=write_results(tmp_path/'a.h5',capsule())
    token=Event();token.set()
    with open_results(path) as reader:
        with pytest.raises(CancelledError):reader.read_field('c0','stress','axial_stress',selection=[0],cancel_event=token)

@pytest.mark.parametrize('fault',['version','missing_input','namespace_escape'])
def test_malformed_role_keeps_previous_destination(tmp_path,fault):
    c=capsule(9,True);path=tmp_path/'a.h5';path.write_bytes(b'previous destination')
    if fault=='version':c['storage_prediction']['version']=7
    elif fault=='missing_input':c['storage_prediction']['derived']['von_mises']['inputs']['axial_stress']='absent'
    else:c['descriptors'][0]['availability']='reconstructable';c['responses']['axial_stress']=c['storage_prediction']['responses']['axial_stress']
    with pytest.raises(ValidationError):write_results(path,c)
    assert path.read_bytes()==b'previous destination'

def test_overflow_durable_zero_prediction_branch_is_authoritative(tmp_path):
    c=capsule();c['storage_prediction']['responses']['axial_stress'][:,1:]=1.e308
    c['coefficients'][:]=4.
    path=write_results(tmp_path/'overflow.h5',c)
    with h5py.File(path) as file:
        assert (file['components/0/prediction_modes'][()]==1).all()
    with open_results(path) as reader:
        for index in range(2):assert_bits(reader.read_field(f'c{index}','stress','axial_stress'),c['stored']['axial_stress']['values'][index])

def test_encoded_role_cannot_be_withdrawn_and_keeps_destination(tmp_path):
    c=capsule();path=tmp_path/'previous.h5';path.write_bytes(b'previous')
    with pytest.raises(ValidationError,match='withdraw'),ArchiveSink(path) as sink:
        header={k:v for k,v in c.items() if k not in ('cases','coefficients','stored','descriptors')}
        sink.capsule_begin(header)
        sink.capsule_case(c['cases'][0],c['coefficients'][0])
        row=c['stored']['axial_stress'];sink.capsule_stored('axial_stress',np.array([0]),row['values'][:1])
        sink.capsule_end(dict(descriptors=c['descriptors'],associations=c['associations'],responses={},derived={},case_count=1))
    assert path.read_bytes()==b'previous'

def test_cancel_during_integrity_validation_does_not_cache_tile(tmp_path):
    path=write_results(tmp_path/'cancel.h5',capsule(32769))
    class Token:
        calls=0
        def is_set(self):
            self.calls+=1
            return self.calls>=5
    with open_results(path) as reader:
        with pytest.raises(CancelledError):reader.read_field('c0','stress','axial_stress',selection=[0],cancel_event=Token())
        assert reader.diagnostics['storage_prediction_tile_decodes']==0

@pytest.mark.parametrize('stream',[False,True])
def test_wide_dependency_closure_uses_bounded_integrity_tiles(tmp_path,stream):
    c=capsule(32769,True)
    for name,values in list(c['storage_prediction']['responses'].items()):
        enlarged=np.zeros((len(values),151),dtype=np.float64);enlarged[:,:3]=values
        c['storage_prediction']['responses'][name]=enlarged
    c['patterns']=tuple(f'p{i}' for i in range(150))
    weights=np.zeros((2,150));weights[:,:2]=c['coefficients'];c['coefficients']=weights
    path=tmp_path/'wide.h5'
    if stream:
        header={k:v for k,v in c.items() if k not in ('cases','coefficients','stored','descriptors')}
        with ArchiveSink(path) as sink:
            sink.capsule_begin(header)
            for record,row in zip(c['cases'],weights):sink.capsule_case(record,row)
            for key,row in c['stored'].items():sink.capsule_stored(key,row['case_indices'],row['values'])
            sink.capsule_end(dict(descriptors=c['descriptors'],associations=c['associations'],responses={},derived={},case_count=2,storage_prediction=c['storage_prediction']))
    else:write_results(path,c)
    info=prediction.manifest(c['storage_prediction'],c['patterns'])
    with h5py.File(path) as file:
        metadata=__import__('json').loads(file['metadata_json'][()])
        for key in ('von_mises','equivalent_stress'):
            step=file['components'][metadata['component_groups'][key]]['values'].chunks[1]
            assert 1<=step<32768 and prediction.work_bytes(info,key,step,151)<=prediction.WORK_BYTES//2
    with open_results(path) as reader:
        assert_bits(reader.read_field('c1','stress','equivalent_stress'),c['stored']['equivalent_stress']['values'][1])
        assert reader.diagnostics['working_peak_bytes']<=prediction.WORK_BYTES

@pytest.mark.parametrize('cache_bytes',[0,8*1024**2])
def test_primitive_basis_reuse_and_all_active_dependencies_accounted(tmp_path,monkeypatch,cache_bytes):
    c=capsule(32769,True);original=h5py.Dataset.__getitem__;reads=[]
    def observed(self,index):
        if self.name.startswith('/storage_prediction/') and self.name.endswith('/responses'):reads.append(self.name)
        return original(self,index)
    monkeypatch.setattr(h5py.Dataset,'__getitem__',observed)
    path=write_results(tmp_path/'a.h5',c)
    # Every component invokes its own bounded write event; each primitive
    # dependency is fetched once/tile, not once/case (two cases in fixture).
    info=prediction.manifest(c['storage_prediction'],c['patterns'])
    expected=0
    with h5py.File(path) as file:
        metadata=__import__('json').loads(file['metadata_json'][()])
        for key in info['targets']:
            step=file['components'][metadata['component_groups'][key]]['values'].chunks[1]
            expected+=((32769+step-1)//step)*len(prediction.primitives(info,key))
    assert len(reads)==expected
    with open_results(path,cache_bytes=cache_bytes) as reader:
        reads.clear();reader.read_field('c0','stress','von_mises',selection=[0]);first=len(reads)
        assert first==6
        reader.read_field('c1','stress','von_mises',selection=[0])
        assert len(reads)==(6 if cache_bytes else 12)
        assert reader.diagnostics['cache_resident_bytes']<=cache_bytes
        reader.clear_cache();assert reader.diagnostics['cache_resident_bytes']==0
        reader.read_field('c1','stress','von_mises',selection=[1]);assert len(reads)==(12 if cache_bytes else 18)

@pytest.mark.parametrize('fault',['missing_descriptor','missing_source_association','missing_count'])
def test_malformed_storage_namespace_is_typed_before_hdf_lookup(tmp_path,fault):
    c=capsule();path=tmp_path/'previous.h5';path.write_bytes(b'previous')
    if fault=='missing_descriptor':
        role=c['storage_prediction'];role['responses']['unbound']=role['responses']['axial_stress'];role['associations']['unbound']=role['associations']['axial_stress']
    elif fault=='missing_source_association':del c['associations']['axial_stress']
    else:c['storage_prediction']['responses']['axial_stress']=np.empty((3,3));c['storage_prediction']['associations']['axial_stress']=np.arange(3)
    with pytest.raises(ValidationError):write_results(path,c)
    assert path.read_bytes()==b'previous'

def test_one_row_fixed_prediction_work_refuses_without_allocation():
    c=capsule(9,True);info=prediction.manifest(c['storage_prediction'],c['patterns'])
    with pytest.raises(WorkingSetError):prediction.tile_rows(info,'equivalent_stress',1,10**9)



def _publish_association_fixture(path, c, stream):
    if not stream:
        return write_results(path, c)
    header={k:v for k,v in c.items() if k not in ('cases','coefficients','stored','descriptors')}
    with ArchiveSink(path) as sink:
        sink.capsule_begin(header)
        for record,row in zip(c['cases'],c['coefficients']):sink.capsule_case(record,row)
        for key,row in c['stored'].items():sink.capsule_stored(key,row['case_indices'],row['values'])
        sink.capsule_end(dict(descriptors=c['descriptors'],associations=c['associations'],responses={},derived={},case_count=2,storage_prediction=c['storage_prediction']))
    return path

@pytest.mark.parametrize('stream',[False,True])
def test_association_tables_physically_shared_between_components_and_role(tmp_path,stream):
    c=capsule(17,True);path=_publish_association_fixture(tmp_path/'shared.h5',c,stream)
    with h5py.File(path) as file:
        tables=[group['associations'] for root in ('components','storage_prediction') for group in file[root].values()]
        assert len(tables)==16
        addresses={h5py.h5o.get_info(ds.id).addr for ds in tables}
        assert len(addresses)==1
        assert sum({h5py.h5o.get_info(ds.id).addr:ds.id.get_storage_size() for ds in tables}.values())==tables[0].id.get_storage_size()>0
        for ds in tables:
            assert ds.dtype==np.dtype('int64') and ds.chunks==(17,) and ds.fletcher32 and ds.compression=='lzf'
            assert np.array_equal(ds[:],c['associations']['axial_stress'])
    with open_results(path) as reader:
        assert_bits(reader.read_field('c1','stress','equivalent_stress'),c['stored']['equivalent_stress']['values'][1])

@pytest.mark.parametrize('stream',[False,True])
def test_shared_association_still_checks_independent_role_digest(tmp_path,stream):
    path=_publish_association_fixture(tmp_path/'shared-corrupt.h5',capsule(17,True),stream)
    with h5py.File(path,'r+') as file:
        ds=next(iter(file['storage_prediction'].values()))['associations']
        ds[0]=999999
    with pytest.raises(ValidationError,match='association'):
        with open_results(path):pass

def test_association_pool_preserves_shape_order_and_distinct_tables(tmp_path):
    from anypost.archive import _AssociationPool
    with h5py.File(tmp_path/'pool.h5','w') as file:
        pool=_AssociationPool();a=np.array([101,103,105],dtype=np.int64)
        first=pool(file,'associations',a,None,integer=True)
        same=pool(file.create_group('same'),'associations',a.astype(np.int32),None,integer=True)
        changed=pool(file.create_group('changed'),'associations',a[::-1],None,integer=True)
        shaped=pool(file.create_group('shaped'),'associations',a[:,None],None,integer=True)
        assert first.id==same.id and first.id!=changed.id and first.id!=shaped.id
        assert np.array_equal(changed[:],a[::-1]) and shaped.shape==(3,1)

def test_stream_empty_shared_role_withdrawal_and_interruption_preserve_destination(tmp_path):
    c=capsule(9,True);header={k:v for k,v in c.items() if k not in ('cases','coefficients','stored','descriptors')}
    # An unused source role has no advertised scientific association tables.
    header['associations']={}
    path=tmp_path/'empty.h5'
    with ArchiveSink(path) as sink:
        sink.capsule_begin(header)
        sink.capsule_end(dict(descriptors=[],associations={},responses={},derived={},case_count=0))
    with h5py.File(path) as file:
        assert file.attrs['complete'] and 'storage_prediction' not in file and len(file['components'])==0
    previous=path.read_bytes()
    with pytest.raises(RuntimeError,match='interrupt'),ArchiveSink(path) as sink:
        sink.capsule_begin(header)
        raise RuntimeError('interrupt before any role rows')
    assert path.read_bytes()==previous and sink.diagnostic_path.exists()
