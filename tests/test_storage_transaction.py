"""Production writer cancellation, cumulative read work and hardlink refusal."""
from copy import deepcopy
from threading import Event
import json
import h5py
import numpy as np
import pytest
from anypost import write_results,open_results,ArchiveSink,RequestCancelled,ValidationError,WorkingSetError
from anypost import storage_prediction as prediction
from test_storage_prediction import capsule,assert_bits

@pytest.mark.parametrize('stream',[False,True])
def test_cancel_after_first_actual_encoder_mask_preserves_destination(tmp_path,monkeypatch,stream):
    c=capsule(32769);path=tmp_path/'previous.h5';path.write_bytes(b'previous');event=Event();writes=[]
    original=h5py.Dataset.__setitem__
    def record(self,index,value):
        result=original(self,index,value)
        if self.name.endswith('/values') and self.dtype==np.dtype('uint64'):
            writes.append((self.name,index));event.set()
        return result
    monkeypatch.setattr(h5py.Dataset,'__setitem__',record)
    with pytest.raises(RequestCancelled):
        if stream:
            with ArchiveSink(path,cancel_event=event) as sink:
                sink.capsule_begin({k:v for k,v in c.items() if k not in ('cases','coefficients','stored','descriptors')})
                for case,weights in zip(c['cases'],c['coefficients']):sink.capsule_case(case,weights)
                row=c['stored']['axial_stress'];sink.capsule_stored('axial_stress',row['case_indices'],row['values'])
        else:write_results(path,c,cancel_event=event)
    assert len(writes)==1 and path.read_bytes()==b'previous'
    assert not list(tmp_path.glob('*.tmp')) and not list(tmp_path.glob('.*.tmp'))
    if stream:assert json.loads(sink.diagnostic_path.read_text())['status']=='cancelled'

@pytest.mark.parametrize('stream',[False,True])
def test_cancellation_api_exclusivity_and_precancel_before_file_mutation(tmp_path,stream):
    event=Event();event.set();path=tmp_path/'absent'/'file.h5'
    with pytest.raises(TypeError,match='not both'):
        if stream:ArchiveSink(path,cancel_event=event,cancel_check=lambda:None)
        else:write_results(path,capsule(),cancel_event=event,cancel_check=lambda:None)
    assert not path.parent.exists()
    with pytest.raises(RequestCancelled):
        if stream:
            with ArchiveSink(path,cancel_event=event) as sink:sink.capsule_begin({})
        else:write_results(path,capsule(),cancel_event=event)
    # Stream may persist its terminal diagnostic, but no HDF5 can be published.
    assert not path.exists()

@pytest.mark.parametrize('stream',[False,True])
def test_owner_cancel_check_exception_identity_survives(tmp_path,stream):
    class OwnerCancelled(Exception):pass
    error=OwnerCancelled('owner cancellation');path=tmp_path/'previous.h5';path.write_bytes(b'previous')
    def check():raise error
    with pytest.raises(OwnerCancelled) as info:
        if stream:
            with ArchiveSink(path,cancel_check=check) as sink:sink.capsule_begin({})
        else:write_results(path,capsule(),cancel_check=check)
    assert info.value is error and path.read_bytes()==b'previous'

def test_source_token_callback_preserves_solve_cancelled(tmp_path):
    # Actual public source token ABI; no model, preparation or recovery runs.
    from anysolver.control import CancellationToken,SolveCancelled
    token=CancellationToken();token.cancel('caller source cancellation')
    path=tmp_path/'previous.h5';path.write_bytes(b'previous')
    with pytest.raises(SolveCancelled,match='caller source cancellation'):
        write_results(path,capsule(),cancel_check=token.raise_if_cancelled)
    assert path.read_bytes()==b'previous'

@pytest.mark.parametrize('stream',[False,True])
def test_final_publication_safe_point_preserves_previous_destination(tmp_path,monkeypatch,stream):
    import os
    c=capsule();event=Event();path=tmp_path/'previous.h5';path.write_bytes(b'previous')
    original=os.fsync
    def fsync(fd):result=original(fd);event.set();return result
    monkeypatch.setattr(os,'fsync',fsync)
    with pytest.raises(RequestCancelled):
        if stream:
            with ArchiveSink(path,cancel_event=event) as sink:
                sink.capsule_begin({k:v for k,v in c.items() if k not in ('cases','coefficients','stored','descriptors')})
                for case,weights in zip(c['cases'],c['coefficients']):sink.capsule_case(case,weights)
                row=c['stored']['axial_stress'];sink.capsule_stored('axial_stress',row['case_indices'],row['values'])
                sink.capsule_end(dict(descriptors=c['descriptors'],associations=c['associations'],responses={},derived={},case_count=2,storage_prediction=c['storage_prediction']))
        else:write_results(path,c,cancel_event=event)
    assert path.read_bytes()==b'previous'
    if stream:assert json.loads(sink.diagnostic_path.read_text())['status']=='cancelled'

@pytest.mark.parametrize('stream',[False,True])
def test_actual_writer_encoder_work_is_observed_not_a_configurable_cap(tmp_path,stream):
    c=capsule(32769);path=tmp_path/'work.h5'
    if stream:
        with ArchiveSink(path) as sink:
            sink.capsule_begin({k:v for k,v in c.items() if k not in ('cases','coefficients','stored','descriptors')})
            for case,weights in zip(c['cases'],c['coefficients']):sink.capsule_case(case,weights)
            row=c['stored']['axial_stress'];sink.capsule_stored('axial_stress',row['case_indices'],row['values'])
            sink.capsule_end(dict(descriptors=c['descriptors'],associations=c['associations'],responses={},derived={},case_count=2,storage_prediction=c['storage_prediction']))
    else:write_results(path,c)
    with h5py.File(path) as file:
        peak=int(file.attrs['peak_encode_work_bytes']);step=file['components/0/values'].chunks[1]
        assert peak==prediction.work_bytes(prediction.manifest(c['storage_prediction'],c['patterns']),'axial_stress',step,3)
        assert 0<peak<=prediction.WORK_BYTES
    if stream:assert sink.diagnostics['peak_encode_work_bytes']==peak

@pytest.mark.parametrize('cache_bytes',[0,8*1024**2])
def test_full_integrity_tile_and_duplicate_scatter_cumulative_lowcap(tmp_path,cache_bytes):
    c=capsule(32769);path=write_results(tmp_path/'ledger.h5',c)
    selection=np.tile(np.array([0,1,1,32767],dtype=np.int64),50);n=len(selection)
    info=prediction.manifest(c['storage_prediction'],c['patterns'])
    with h5py.File(path) as file:step=file['components/0/values'].chunks[1]
    work=prediction.work_bytes(info,'axial_stress',step,3);outer=n*24+2*32
    old_cap=outer+n*24+work+1
    with open_results(path,working_bytes=old_cap,cache_bytes=cache_bytes) as reader:
        with pytest.raises(WorkingSetError):reader.read_field('c0','stress','axial_stress',selection=selection)
        assert reader.diagnostics['storage_prediction_tile_decodes']==reader.diagnostics['storage_prediction_basis_reads']==0
    cap=outer+n*64+work
    with open_results(path,working_bytes=cap,cache_bytes=cache_bytes) as reader:
        for selected in (selection,selection[::-1]):
            assert_bits(reader.read_field('c0','stress','axial_stress',selection=selected),c['stored']['axial_stress']['values'][0,selected])
        assert reader.diagnostics['working_peak_bytes']<=cap
        assert reader.diagnostics['cache_resident_bytes']<=cache_bytes
        assert reader.diagnostics['storage_prediction_tile_decodes']==(2 if cache_bytes==0 else 1)

def test_shared_header_association_final_conflict_refuses_without_rewrite(tmp_path):
    c=capsule(9,True);path=write_results(tmp_path/'previous.h5',c);previous=path.read_bytes()
    changed={key:value.copy() for key,value in c['associations'].items()};changed['shear_stress_y'][0]=9999
    with pytest.raises(ValidationError,match='static values changed'),ArchiveSink(path) as sink:
        sink.capsule_begin({k:v for k,v in c.items() if k not in ('cases','coefficients','stored','descriptors')})
        groups=list(sink._handle['components'].values())
        assert groups[0]['associations'].id==groups[1]['associations'].id
        for case,weights in zip(c['cases'],c['coefficients']):sink.capsule_case(case,weights)
        sink.capsule_end(dict(descriptors=c['descriptors'],associations=changed,responses={},derived={},case_count=2,storage_prediction=c['storage_prediction']))
    assert path.read_bytes()==previous
    with open_results(path) as reader:assert np.array_equal(reader.associations('stress','axial_stress'),c['associations']['axial_stress'])
