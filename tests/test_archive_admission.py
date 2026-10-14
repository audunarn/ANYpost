import json
from concurrent.futures import CancelledError
from threading import Event
from types import SimpleNamespace
import h5py
import numpy as np
import pytest
from anypost import (write_results,open_results,ArchiveSink,ValidationError,UnknownCaseError,UnavailableFieldError,FailedCaseError,RuntimeDependencyError)
from test_stored_correction import capsule,bits_equal,stream


def native2(c=None):
    c=capsule(native=False) if c is None else c
    cases=tuple(dict(case,portable=case.get('portable',case['certified']),displacement_portable=case.get('displacement_portable',True if case['certified'] else None)) for case in c['cases'])
    return dict(c,version=2,runtime='anysolver-results/2',combination='ordered_float64/1',cases=cases)


@pytest.mark.parametrize('version,runtime,combination',[(True,'anysolver-results/1',None),(1,'anysolver-results/2',None),(2,'anysolver-results/1','ordered_float64/1'),(2,'anysolver-results/2',None),(1,'anysolver-results/1','ordered_float64/1'),(3,'anysolver-results/3','ordered_float64/1')])
def test_exact_paired_contract_atomic_refusal(tmp_path,version,runtime,combination):
    path=tmp_path/'old.h5';path.write_bytes(b'previous')
    c=dict(capsule(),version=version,runtime=runtime)
    if combination is not None:c['combination']=combination
    with pytest.raises(ValidationError):write_results(path,c)
    assert path.read_bytes()==b'previous'
    with pytest.raises(ValidationError):stream(tmp_path/'stream.h5',c)
    assert not (tmp_path/'stream.h5').exists()


def test_source_pair_preserved_in_native_pool_versions_and_stream(tmp_path):
    c=native2();c['cases']=tuple(dict(case,reaction_node_ids=list(range(300))) for case in c['cases'])
    for writer in (write_results,stream):
        path=tmp_path/(writer.__name__+'.h5');writer(path,c)
        with h5py.File(path) as h:assert h.attrs['version']==2
        with open_results(path) as reader:
            m=reader.metadata();assert (m['capsule_version'],m['runtime'],m['combination'])==(2,'anysolver-results/2','ordered_float64/1')
            bits_equal(reader.read_field('c1','stress','x',[6,0,6]),c['stored']['sx']['values'][1,[6,0,6]])


@pytest.mark.parametrize('flag,value',[('portable',False),('displacement_portable',False),('displacement_portable',None)])
def test_case_flags_prevent_missing_row_fabrication_even_empty(tmp_path,flag,value):
    c=capsule(native=False);c['stored']={}
    c['cases']=tuple(dict(case,**{flag:value}) for case in c['cases'])
    if flag=='displacement_portable':c['descriptors']=(dict(c['descriptors'][0],quantity_id='displacement'),)
    path=write_results(tmp_path/'missing.h5',c)
    with open_results(path) as reader:
        assert reader.case_components('c0')[0]['availability']=='unavailable'
        assert reader.diagnostics['reads']==0 and reader._runtime is None
        for selection in ([],None):
            with pytest.raises(UnavailableFieldError):reader.read_field('c0',c['descriptors'][0]['quantity_id'],'x',selection)
        assert reader.diagnostics['reads']==0


def test_explicit_values_override_nonportable_but_unavailable_always_refuses(tmp_path):
    c=capsule();c['cases']=tuple(dict(case,portable=False,displacement_portable=None) for case in c['cases'])
    path=write_results(tmp_path/'stored.h5',c)
    with open_results(path) as reader:
        assert reader.case_components('c0')[0]['availability']=='stored'
        bits_equal(reader.read_field('c0','stress','x'),c['stored']['sx']['values'][0])
    c['cases']=tuple(dict(case,unavailable_components=['sx']) for case in c['cases'])
    path=write_results(tmp_path/'unavailable.h5',c)
    with open_results(path) as reader:
        assert reader.case_components('c0')[0]['availability']=='unavailable'
        with pytest.raises(UnavailableFieldError):reader.read_field('c0','stress','x',[])
        with pytest.raises(UnavailableFieldError):reader.read_envelope('stress','x',case_ids=['c0'],selection=[])


@pytest.mark.parametrize('update',[{'portable':None},{'portable':1},{'displacement_portable':1},{'unavailable_components':'sx'},{'unavailable_components':['sx','sx']},{'unavailable_components':['missing']}])
def test_case_flags_known_keys_typed_validation_in_writer_and_stream(tmp_path,update):
    c=capsule();c['cases']=tuple(dict(case,**update) for case in c['cases'])
    for writer in (write_results,stream):
        with pytest.raises(ValidationError):writer(tmp_path/(writer.__name__+'.h5'),c)


def test_unknown_failed_discovery_and_envelope_coverage_float64_ties(tmp_path):
    c=capsule(native=False);c['descriptors']=(dict(c['descriptors'][0],tie_policy='exact_earliest_input_index'),)
    c['cases']=tuple(dict(case,input_index=1 if i==0 else 0 if i==1 else i) for i,case in enumerate(c['cases']))
    c['stored']['sx']['values'][0]=1;c['stored']['sx']['values'][1]=1
    c['stored']['sx']['values'][1,0]=np.nextafter(1.,np.inf);c['stored']['sx']['values'][5]=1e99
    path=write_results(tmp_path/'envelope.h5',c)
    with open_results(path) as reader:
        envelope=reader.read_envelope('stress','x',case_ids=['c0','c5','c1'],selection=[0,1,1])
        assert envelope.requested_case_ids==('c0','c5','c1') and envelope.failed_case_ids==envelope.excluded_case_ids==('c5',)
        assert envelope.admitted_case_ids==('c0','c1') and envelope.max_case_ids.tolist()==['c1','c1','c1']
        assert envelope.min_case_ids.tolist()==['c0','c1','c1'] and envelope.maximum[0]==np.nextafter(1.,np.inf)
        with pytest.raises(FailedCaseError) as error:reader.read_envelope('stress','x',case_ids=['c5'],selection=[])
        assert error.value.failed_case_ids==('c5',)
        with pytest.raises(FailedCaseError):reader.read_envelope('stress','x',strict_failed=True)
        with pytest.raises(FailedCaseError):reader.case_components('c5')
        for operation in (lambda:reader.case_metadata('unknown'),lambda:reader.case_components('unknown'),lambda:reader.read_field('unknown','stress','x',[]),lambda:reader.read_envelope('stress','x',case_ids=['unknown'])):
            with pytest.raises(UnknownCaseError) as error:operation()
            assert isinstance(error.value,(ValidationError,KeyError))


def test_cancel_between_tiles_never_publishes_partial_field_and_reusable(tmp_path,monkeypatch):
    c=capsule(7000);path=write_results(tmp_path/'tiles.h5',c);cancel=Event()
    with open_results(path,cache_bytes=4096,working_bytes=65536) as reader:
        original=reader._spatial_read;reads=[]
        def read(ds,indices,row=None):
            result=original(ds,indices,row=row)
            if ds.name.endswith('/values'):
                reads.append(len(indices))
                if len(reads)==2:cancel.set()
            return result
        monkeypatch.setattr(reader,'_spatial_read',read)
        with pytest.raises(CancelledError):reader.read_field('c0','stress','x',slice(0,None,7),cancel_event=cancel)
        assert len(reads)==2 and sum(reads)<1000
        assert not any(key[0]=='field' for key in reader._cache._values)
        assert reader._cancel_event is None and reader.diagnostics['cache_resident_bytes']<=4096
        cancel.clear();bits_equal(reader.read_field('c0','stress','x',[0,7,0]),c['stored']['sx']['values'][0,[0,7,0]])
        cancel.set()
        for operation in (lambda:reader.read_field('c0','stress','x',[],cancel_event=cancel),lambda:reader.associations('stress','x',[],cancel_event=cancel),lambda:reader.read_envelope('stress','x',selection=[],cancel_event=cancel)):
            with pytest.raises(CancelledError):operation()


def test_cancel_whole_source_proof_without_completed_proof_marker(tmp_path,monkeypatch):
    c=capsule(14000);path=write_results(tmp_path/'proof.h5',c);cancel=Event()
    with open_results(path,cache_bytes=0,working_bytes=8192) as reader:
        original=reader._spatial_read;calls=[]
        def read(ds,indices,row=None):
            result=original(ds,indices,row=row)
            if ds.name.endswith('/associations'):
                calls.append(1)
                if len(calls)==2:cancel.set()
            return result
        monkeypatch.setattr(reader,'_spatial_read',read)
        with pytest.raises(CancelledError):reader.read_field('c0','stress','x',[0],cancel_event=cancel)
        assert len(calls)==2 and not reader._verified_codecs
        cancel.clear();bits_equal(reader.read_field('c0','stress','x',[0]),c['stored']['sx']['values'][0,[0]])


@pytest.mark.parametrize('mutation',['no_ids','negative','too_large','float','wrong_count','topology'])
def test_new_geometry_maps_refuse_invalid_source_and_archive(tmp_path,mutation):
    c=native2();c['geometry']=dict(positions=np.zeros((3,3)),lines=np.array([[0,1],[1,2]]),line_to_element=np.array([0,1]),element_ids=np.array([100,300]))
    good=write_results(tmp_path/'good.h5',c)
    g=c['geometry']
    if mutation=='no_ids':del g['element_ids']
    elif mutation=='negative':g['line_to_element'][0]=-1
    elif mutation=='too_large':g['line_to_element'][0]=2
    elif mutation=='float':g['line_to_element']=np.array([0.,1.])
    elif mutation=='wrong_count':g['line_to_element']=np.array([0])
    else:g['lines'][0,0]=3
    with pytest.raises(ValidationError):write_results(tmp_path/'bad.h5',c)
    with h5py.File(good,'r+') as h:
        geometry=h['geometry'];del geometry['line_to_element'];geometry.create_dataset('line_to_element',data=np.array([-1,0]))
    with pytest.raises(ValidationError):open_results(good)


def test_ordered_runtime_capability_and_prepared_tag_dispatch(tmp_path,monkeypatch):
    import anysolver_results as old
    import anypost.archive as archive
    c=native2();c['stored']={};path=write_results(tmp_path/'runtime.h5',c);calls=[]
    def prepare(basis,*,active_indices=None,combination):
        calls.append(combination)
        raw=old.prepare_response(basis,active_indices=active_indices)
        return SimpleNamespace(raw=raw,nbytes=raw.nbytes,combination=combination)
    def replay(prepared,weights,*,combination):
        assert combination==prepared.combination
        return old.reconstruct_prepared_response(prepared.raw,weights)
    runtime=SimpleNamespace(RUNTIME_ID='anysolver-results/2',SUPPORTED_RUNTIME_IDS=('anysolver-results/1','anysolver-results/2'),SUPPORTED_COMBINATIONS=('legacy_blas/1','ordered_float64/1'),prepare_response=prepare,reconstruct_prepared_response=replay)
    monkeypatch.setattr(archive.importlib,'import_module',lambda name:runtime)
    with open_results(path) as reader:
        reader.read_field('c0','stress','x',[0,2,0]);assert calls==['ordered_float64/1']
    runtime.SUPPORTED_COMBINATIONS=()
    with open_results(path) as reader:
        with pytest.raises(RuntimeDependencyError):reader.read_field('c0','stress','x',[])
        assert reader.diagnostics['reads']==0


def test_real_reader_latest_cancel_close_no_partial_cache_and_collectable(tmp_path,monkeypatch):
    import gc,weakref
    from time import monotonic,sleep
    from queue import Queue
    from anypost import LatestRequestLoader
    c=capsule(7000);path=write_results(tmp_path/'latest.h5',c)
    reader=open_results(path,cache_bytes=4096,working_bytes=65536)
    original=reader._spatial_read;entered=Event();release=Event();queue=Queue();seen=[];calls=[]
    def read(ds,indices,row=None):
        result=original(ds,indices,row=row)
        if ds.name.endswith('/values'):
            calls.append(1)
            if len(calls)==1:entered.set();assert release.wait(2)
        return result
    monkeypatch.setattr(reader,'_spatial_read',read)
    loader=LatestRequestLoader(queue.put)
    loader.request(lambda cancel:reader.read_field('c0','stress','x',slice(0,None,7),cancel_event=cancel),seen.append)
    assert entered.wait(2)
    loader.request(lambda cancel:reader.read_field('c1','stress','x',[0],cancel_event=cancel),seen.append)
    release.set();deadline=monotonic()+3
    while queue.empty():assert monotonic()<deadline;sleep(.002)
    while not queue.empty():queue.get_nowait()()
    deadline=monotonic()+3
    while not seen:
        assert monotonic()<deadline
        while not queue.empty():queue.get_nowait()()
        sleep(.002)
    assert len(seen)==1 and seen[0].error is None
    bits_equal(seen[0].value,c['stored']['sx']['values'][1,[0]])
    assert not any(key[0]=='field' and key[3]=='c0' for key in reader._cache._values)
    loader.close();reader.close();assert reader._closed and reader.diagnostics['cache_resident_bytes']==0
    reference=weakref.ref(loader);del loader;gc.collect();assert reference() is None


def test_legacy_case_discovery_and_cancel_keyword_compatibility(tmp_path):
    from test_legacy import legacy_fixture
    path=tmp_path/'legacy.h5';legacy_fixture(path,samples=True)
    cancel=Event();cancel.set()
    with open_results(path) as reader:
        assert reader.case_components('0')[0]['availability']=='stored'
        with pytest.raises(CancelledError):reader.read_field('0','stress','eq',cancel_event=cancel)
        with pytest.raises(CancelledError):reader.associations('stress','eq',cancel_event=cancel)
        cancel.clear();assert reader.associations('stress','eq',cancel_event=cancel).shape==(6,2)


@pytest.mark.parametrize('update',[{'capsule_version':True},{'capsule_version':2},{'runtime':'anysolver-results/2'},{'combination':'ordered_float64/1'}])
def test_native_metadata_pair_corruption_typed_refused(tmp_path,update):
    path=write_results(tmp_path/'pair.h5',capsule())
    with h5py.File(path,'r+') as h:
        metadata=json.loads(h['metadata_json'][()]);metadata.update(update)
        del h['metadata_json'];h.create_dataset('metadata_json',data=json.dumps(metadata),dtype=h5py.string_dtype())
    with pytest.raises(ValidationError):open_results(path)


def test_old_native_metadata_pair_defaults_retain_legacy_meaning(tmp_path):
    path=write_results(tmp_path/'old.h5',capsule())
    with h5py.File(path,'r+') as h:
        metadata=json.loads(h['metadata_json'][()]);metadata.pop('capsule_version');metadata.pop('combination')
        del h['metadata_json'];h.create_dataset('metadata_json',data=json.dumps(metadata),dtype=h5py.string_dtype())
    with open_results(path) as reader:
        assert 'capsule_version' not in reader.metadata()
        bits_equal(reader.read_field('c0','stress','x',[0]),capsule()['stored']['sx']['values'][0,[0]])


def test_missing_dependency_case_admission_precedes_runtime_and_empty_allocation(tmp_path,monkeypatch):
    c=capsule(native=False);c['stored']={}
    c['descriptors']=(c['descriptors'][0],dict(c['descriptors'][0],key='other',component='other'))
    c['associations']['other']=c['associations']['sx'];c['derived']={'other':{'operation':'test_not_called','inputs':{'x':'sx'}}}
    c['cases']=tuple(dict(case,unavailable_components=['sx']) for case in c['cases'])
    path=write_results(tmp_path/'dependent.h5',c)
    with open_results(path) as reader:
        assert [d['availability'] for d in reader.case_components('c0')]==['unavailable','unavailable']
        monkeypatch.setattr('anypost.archive.importlib.import_module',lambda name: (_ for _ in ()).throw(AssertionError('unavailable dependency imported runtime')))
        with pytest.raises(UnavailableFieldError):reader.read_field('c0','stress','other',[])
        assert reader.diagnostics['reads']==0


@pytest.mark.parametrize("version",[True,False,1.0,3])
def test_native_archive_container_version_typed_not_coerced(tmp_path,version):
    path=write_results(tmp_path/"container.h5",capsule())
    with h5py.File(path,"r+") as h:h.attrs["version"]=version
    with pytest.raises(ValidationError):open_results(path)
