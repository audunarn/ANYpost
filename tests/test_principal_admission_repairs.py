import json
import os
from pathlib import Path
import h5py
import numpy as np
import pytest
from anypost import write_results,open_results,ValidationError,RecoveryError
from test_nodal_replay_admission import nodal,stream
from test_archive_admission import native2
from test_derived_caching import shell_capsule


@pytest.mark.parametrize('location',['Node','nodes','point','element'])
@pytest.mark.parametrize('writer',[write_results,stream])
def test_forged_nodal_alias_location_refuses_write_atomically(tmp_path,location,writer):
    c=nodal('translation','UX','translation:UX');c['descriptors']=(dict(c['descriptors'][0],location=location),)
    c['cases']=tuple(dict(case,displacement_portable=False) for case in c['cases'])
    path=tmp_path/'old.h5';path.write_bytes(b'previous')
    with pytest.raises(ValidationError,match='canonical'):writer(path,c)
    assert path.read_bytes()==b'previous'


@pytest.mark.parametrize('location',['Node','element'])
def test_forged_alias_native_open_refuses_before_runtime_import(tmp_path,location,monkeypatch):
    path=write_results(tmp_path/'valid.h5',nodal())
    with h5py.File(path,'r+') as h:
        meta=json.loads(h['metadata_json'][()]);meta['descriptors'][0].update(key='translation:UX',quantity_id='translation',component='UX',location=location)
        meta['cases'][0]['displacement_portable']=False
        del h['metadata_json'];h.create_dataset('metadata_json',data=json.dumps(meta),dtype=h5py.string_dtype())
    monkeypatch.setattr('anypost.archive.importlib.import_module',lambda *_:pytest.fail('admission refusal precedes runtime'))
    with pytest.raises(ValidationError,match='canonical'):open_results(path)


@pytest.mark.parametrize('quantity,key,location',[('displacement','displacement:UX','element'),('displacement','foo','element'),('reaction','reaction:FX','Node')])
def test_reserved_nodal_wire_classes_cannot_claim_element(tmp_path,quantity,key,location):
    c=nodal(quantity,'UX' if quantity=='displacement' else 'FX',key);c['descriptors']=(dict(c['descriptors'][0],location=location),)
    with pytest.raises(ValidationError,match='canonical'):write_results(tmp_path/'invalid.h5',c)


def test_old_source_gold_metadata_and_generic_element_stored_labels_remain_valid(tmp_path):
    # These are unchanged original-source fixtures, not regenerated models.
    # The authoritative files remain on C; WSL reads that same drive mount.
    root=Path('C:/ANYworkspaceWork/anypost-20261012/runtime2-linux-evidence/golden') if os.name=='nt' else Path('/mnt/c/ANYworkspaceWork/anypost-20261012/runtime2-linux-evidence/golden')
    for filename in ('beam2.h5','beam3.h5','rotated_shell_pressure.h5'):
        with open_results(root/filename) as reader:assert len(reader.cases())==5 and reader._runtime is None
    c=native2();c['stored']={};c['descriptors']=(dict(c['descriptors'][0],quantity_id='sx',component='sx',location='element'),)
    with open_results(write_results(tmp_path/'generic.h5',c)) as reader:assert reader.components()[0]['quantity_id']=='sx'
    c=nodal('translation','UX','translation:UX',stored=True);c['descriptors']=(dict(c['descriptors'][0],location='Node'),)
    with open_results(write_results(tmp_path/'stored.h5',c)) as reader:np.testing.assert_array_equal(reader.read_field('c0','translation','UX'),c['stored']['translation:UX']['values'][0])
    c=nodal('translation','UX','translation:UX');c.update(version=1,runtime='anysolver-results/1');c.pop('combination');c['descriptors']=(dict(c['descriptors'][0],location='element'),)
    with open_results(write_results(tmp_path/'legacy.h5',c)) as reader:assert reader._runtime is None


@pytest.mark.parametrize('workers',[1,2])
def test_actual_principal_overflow_chains_typed_runtime_and_reader_errors(tmp_path,workers):
    c=shell_capsule(1);c.update(version=2,runtime='anysolver-results/2',combination='ordered_float64/1',patterns=(),coefficients=np.empty((1,0)))
    c['cases']=({'case_id':'a','status':'converged','certified':True,'portable':True,'displacement_portable':True,'route':'basis','active_indices':[]},)
    c['responses']={key:np.array([[1.7e308 if key in ('membrane_xx','bending_xx') else 0.]]) for key in c['responses']}
    path=write_results(tmp_path/'overflow.h5',c)
    with np.errstate(over='ignore'),open_results(path,derivation_workers=workers) as reader:
        with pytest.raises(RecoveryError) as error:reader.read_field('a','stress','principal')
        assert error.value.case_id=='a' and error.value.component_key=='principal'
        import anysolver_results as runtime
        assert isinstance(error.value.__cause__,runtime.DerivationError) and isinstance(error.value.__cause__.__cause__,ValueError)


def test_stress_disguise_with_reserved_dof_component_refuses_write_and_open(tmp_path,monkeypatch):
    c=nodal('stress','UX','translation:UX');c['descriptors']=(dict(c['descriptors'][0],location='element'),)
    c['cases']=tuple(dict(case,displacement_portable=False) for case in c['cases'])
    for writer in (write_results,stream):
        with pytest.raises(ValidationError,match='canonical'):writer(tmp_path/(writer.__name__+'.h5'),c)
    path=write_results(tmp_path/'valid.h5',nodal())
    with h5py.File(path,'r+') as h:
        meta=json.loads(h['metadata_json'][()]);meta['descriptors'][0].update(key='translation:UX',quantity_id='stress',component='UX',location='element');meta['cases'][0]['displacement_portable']=False
        del h['metadata_json'];h.create_dataset('metadata_json',data=json.dumps(meta),dtype=h5py.string_dtype())
    monkeypatch.setattr('anypost.archive.importlib.import_module',lambda *_:pytest.fail('wire admission precedes runtime'))
    with pytest.raises(ValidationError,match='canonical'):open_results(path)


def test_canonical_nodal_class_still_honours_explicit_displacement_refusal(tmp_path,monkeypatch):
    from anypost import UnavailableFieldError
    c=nodal();c['cases']=tuple(dict(case,displacement_portable=False) for case in c['cases'])
    path=write_results(tmp_path/'canonical-refused.h5',c)
    monkeypatch.setattr('anypost.archive.importlib.import_module',lambda *_:pytest.fail('case flag refusal precedes runtime'))
    with open_results(path) as reader:
        assert reader.case_components('c0')[0]['availability']=='unavailable'
        with pytest.raises(UnavailableFieldError):reader.read_field('c0','displacement','UX',[])
        assert reader._runtime is None and reader.diagnostics['reads']==0
