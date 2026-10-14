"""Independent review N1-N4 regressions; no engineering formula changes."""
import json
from concurrent.futures import CancelledError
from threading import Event
import h5py
import numpy as np
import pytest
import anysolver_results as runtime
from anypost import (write_results,open_results,ValidationError,UnavailableFieldError,RecoveryError,WorkingSetError,IdentityError)
from test_archive_admission import native2
from test_stored_correction import capsule,stream,bits_equal


@pytest.mark.parametrize('missing',['portable','displacement_portable'])
def test_capsule2_missing_flag_writer_stream_and_reader_fail_closed(tmp_path,missing,monkeypatch):
    good=native2();path=write_results(tmp_path/'valid.h5',good)
    for writer in (write_results,stream):
        c=native2();c['cases']=tuple(dict(case) for case in c['cases']);c['cases'][0].pop(missing)
        target=tmp_path/(writer.__name__+'.h5');target.write_bytes(b'previous')
        with pytest.raises(ValidationError):writer(target,c)
        assert target.read_bytes()==b'previous'
    with h5py.File(path,'r+') as h:
        meta=json.loads(h['metadata_json'][()]);meta['cases'][0].pop(missing)
        del h['metadata_json'];h.create_dataset('metadata_json',data=json.dumps(meta),dtype=h5py.string_dtype())
    monkeypatch.setattr('anypost.archive.importlib.import_module',lambda name: (_ for _ in ()).throw(AssertionError('missing flag imported runtime')))
    with pytest.raises(ValidationError):open_results(path)


def test_renamed_displacement_quantity_cannot_bypass_source_key_control(tmp_path):
    c=native2(capsule(native=False));c['stored']={}
    c['descriptors']=(dict(c['descriptors'][0],key='displacement:UX',quantity_id='renamed_translation'),)
    c['associations']={'displacement:UX':c['associations']['sx']};c['responses']={'displacement:UX':c['responses']['sx']}
    c['cases']=tuple(dict(case,displacement_portable=None) for case in c['cases'])
    # The stronger source2 wire admission now refuses this descriptor before
    # publication, rather than allowing metadata then refusing its field.
    with pytest.raises(ValidationError,match='canonical'):
        write_results(tmp_path/'renamed.h5',c)


@pytest.mark.parametrize('encoding',['auto','plain'])
def test_explicit_actual_rows_survive_genuine_nonportable_flags(tmp_path,encoding):
    c=native2();c['cases']=tuple(dict(case,portable=False,displacement_portable=None) for case in c['cases'])
    path=write_results(tmp_path/'explicit.h5',c,stored_encoding=encoding)
    with open_results(path) as reader:
        bits_equal(reader.read_field('c0','stress','x',[69,0,69]),c['stored']['sx']['values'][0,[69,0,69]])
        assert reader._runtime is None


@pytest.mark.parametrize('path_kind',['stored_native','stored_flat','prepared','ordinary'])
def test_range_and_three_explicit_ids_never_alias_cache_keys(tmp_path,path_kind,monkeypatch):
    c=native2(capsule(70,native=path_kind=='stored_native'))
    if path_kind in ('prepared','ordinary'):c['stored']={}
    if path_kind=='ordinary':
        monkeypatch.setattr(runtime,'prepare_response',None);monkeypatch.setattr(runtime,'reconstruct_prepared_response',None)
    path=write_results(tmp_path/'selectors.h5',c)
    with open_results(path,working_bytes=65536,cache_bytes=65536) as reader:
        active=c['cases'][0]['active_indices'];weights=c['coefficients'][0]
        expected=c['stored']['sx']['values'][0] if c['stored'] else runtime.reconstruct_response(c['responses']['sx'],weights,active_indices=active,combination='ordered_float64/1')
        for selection in (slice(0,48,3),[0,48,3],slice(0,48,3),[0,48,3],[48,3,48,0],slice(45,None,-3)):
            indices=np.arange(70)[selection]
            bits_equal(reader.read_field('c0','stress','x',selection),expected[indices])
        assert reader.diagnostics['cache_hits']>0
        assert reader.diagnostics['working_peak_bytes']<=65536 and reader.diagnostics['cache_peak_bytes']<=65536


def test_runtime_errors_are_typed_with_case_component_and_original_cause(tmp_path):
    c=native2(capsule(native=False));c['stored']={};c['responses']['sx'][:,1]=1e308;c['coefficients'][0,0]=2.
    path=write_results(tmp_path/'overflow.h5',c)
    with open_results(path) as reader:
        with pytest.raises(RecoveryError) as error:reader.read_field('c0','stress','x',[0])
        assert isinstance(error.value.__cause__,runtime.ReconstructionError)
        assert (error.value.case_id,error.value.component_key,error.value.runtime,error.value.combination)==('c0','sx','anysolver-results/2','ordered_float64/1')
        assert 'c0' in str(error.value) and 'sx' in str(error.value)
    c=native2(capsule(native=False));c['stored']={};c['descriptors']+=(dict(c['descriptors'][0],key='bad',component='bad'),)
    c['associations']['bad']=c['associations']['sx'];c['derived']={'bad':{'operation':'not_registered','inputs':{'x':'sx'}}}
    path=write_results(tmp_path/'derive.h5',c)
    with open_results(path) as reader:
        with pytest.raises(RecoveryError) as error:reader.read_field('c0','stress','bad',[0])
        assert isinstance(error.value.__cause__,runtime.DerivationError) and error.value.component_key=='bad'


@pytest.mark.parametrize('error',[TypeError('programmer'),CancelledError(),WorkingSetError('cap'),IdentityError('identity')])
def test_runtime_boundary_does_not_mask_unrelated_errors(tmp_path,monkeypatch,error):
    c=native2(capsule(native=False));c['stored']={};path=write_results(tmp_path/'error.h5',c)
    def fail(*args,**kwargs):raise error
    monkeypatch.setattr(runtime,'prepare_response',fail)
    with open_results(path) as reader:
        with pytest.raises(type(error)) as caught:reader.read_field('c0','stress','x',[0])
        assert caught.value is error


def test_diagnostics_expose_replay_contract_without_changing_legacy_metadata(tmp_path):
    for c,version,combination,deterministic in ((capsule(),1,'legacy_blas/1',False),(native2(),2,'ordered_float64/1',True)):
        path=write_results(tmp_path/f'v{version}.h5',c)
        with open_results(path) as reader:
            d=reader.diagnostics
            assert (d['capsule_version'],d['runtime'],d['combination'],d['response_layout_deterministic'])==(version,c['runtime'],combination,deterministic)
            assert d['determinism_scope']=='response_combination_only'
    from test_legacy import legacy_fixture
    path=tmp_path/'sidecar.h5';legacy_fixture(path)
    with open_results(path) as reader:
        assert reader.diagnostics['combination'] is None and reader.diagnostics['response_layout_deterministic'] is None


def test_envelope_names_every_unavailable_case_in_source_input_order(tmp_path):
    c=native2();c['cases']=tuple(dict(case,input_index=10-i,unavailable_components=['sx'] if i in (0,2) else []) for i,case in enumerate(c['cases']))
    path=write_results(tmp_path/'coverage.h5',c)
    with open_results(path) as reader:
        with pytest.raises(UnavailableFieldError) as error:reader.read_envelope('stress','x',case_ids=['c0','c5','c1','c2'],selection=[])
        assert error.value.unavailable_case_ids==('c2','c0')
        assert error.value.requested_case_ids==('c0','c5','c1','c2') and error.value.failed_case_ids==('c5',)
        with pytest.raises(AttributeError):error.value.unavailable_case_ids=('modified',)
        assert reader.diagnostics['reads']==0 and reader._runtime is None
