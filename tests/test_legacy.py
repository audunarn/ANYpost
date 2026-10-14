import h5py
import json
import numpy as np
import pytest
from anypost import open_results,FailedCaseError,UnavailableFieldError,IdentityError

def legacy_fixture(path,*,partial=False,samples=False):
    with h5py.File(path,"w") as h:
        h.attrs.update(schema="anyfem.result",version=1,complete=True,partial=partial,model_hash="model",mesh_hash="mesh")
        frame=h.create_group("frames");frame.attrs["kind"]="time"; frame.create_dataset("values",data=[0.,1.])
        fields=h.create_group("fields");g=fields.create_group("stress")
        if samples:
            values=np.arange(12,dtype=float).reshape(2,3,2)
            components=["eq"];provenance={"scalar_sample_axes":[2]}
        else:
            values=np.arange(12,dtype=float).reshape(2,3,2)
            components=["x","y"];provenance={"source":"actual-v1-schema"}
        g.create_dataset("values",data=values)
        g.attrs.update(key="stress",label="Stress",location="element",unit="Pa",basis="local",components=json.dumps(components),frames=json.dumps([0.,1.]),provenance=json.dumps(provenance),recovery="stored")
        tables=h.create_group("tables");tables.create_dataset("stress_element_ids",data=[11,12,13])
        tables.create_dataset("notes",data=json.dumps({"a":1}),dtype=h5py.string_dtype())
        metadata=h.create_group("metadata");metadata.create_dataset("provenance_json",data=json.dumps({"source":"test-v1"}),dtype=h5py.string_dtype())
    return values

def test_legacy_frames_component_tables_and_provenance(tmp_path):
    path=tmp_path/"legacy.h5";values=legacy_fixture(path)
    with pytest.raises(IdentityError):open_results(path,expected_mesh_identity="wrong")
    with open_results(path,expected_model_identity="model") as reader:
        assert reader.diagnostics["reads"]==0
        assert [c["frame"] for c in reader.cases()]==[0.,1.]
        assert reader.components()[0]["basis"]=="local"
        assert reader.metadata()["provenance"]["provenance"]["source"]=="test-v1"
        np.testing.assert_array_equal(reader.read_field("1","stress","y",[2,0,2]),values[1,[2,0,2],1])
        np.testing.assert_array_equal(reader.associations("stress","x"),[11,12,13])
        assert reader.table("notes")=={"a":1}
        assert reader.geometry()=={}
        assert reader.read_envelope("stress","x").max_case_ids.tolist()==["1"]*3

def test_legacy_samples_exact_associations(tmp_path):
    path=tmp_path/"samples.h5";values=legacy_fixture(path,samples=True)
    with open_results(path) as reader:
        np.testing.assert_array_equal(reader.read_field("1","stress","eq"),values[1].ravel())
        np.testing.assert_array_equal(reader.associations("stress","eq"),[[11,0],[11,1],[12,0],[12,1],[13,0],[13,1]])

def test_legacy_partial_and_missing_ids(tmp_path):
    path=tmp_path/"partial.h5";legacy_fixture(path,partial=True)
    with open_results(path) as reader:
        assert reader.cases()[0]["status"]=="partial"
        with pytest.raises(FailedCaseError):reader.read_field("0","stress","x")
    legacy_fixture(path)
    with h5py.File(path,"r+") as h:del h["tables/stress_element_ids"]
    with open_results(path) as reader:
        with pytest.raises(UnavailableFieldError):reader.associations("stress","x")


@pytest.mark.parametrize('axis',[[0.,float('nan')],[float('nan'),float('nan')]])
def test_legacy_nan_full_axes_metadata_accessors_and_ordinal_reads(tmp_path,axis):
    path=tmp_path/'nan_labels.h5';values=legacy_fixture(path)
    with h5py.File(path,'r+') as h:
        h['frames/values'][:]=axis;h['fields/stress'].attrs['frames']=json.dumps(axis)
        h.attrs['legacy_nan']=np.nan;h.attrs['header_vector']=np.array([1.,np.nan])
    with open_results(path) as reader:
        methods=(reader.metadata,reader.metadata_summary,reader.cases,reader.case_summaries,reader.components,lambda:reader.case_metadata('1'))
        for method in methods:assert method() is not None
        assert reader.components()[0]['frame_axis']=='root' and reader.components()[0]['case_ids']==['0','1']
        assert np.isnan(reader.case_metadata('1')['frame']) and np.isnan(reader.cases()[1]['frame'])
        assert np.isnan(reader.case_summaries()[1]['frame'])
        assert reader.metadata_summary()['cases_complete'] is True
        np.testing.assert_array_equal(reader.frames(),axis)
        np.testing.assert_array_equal(reader.read_field('1','stress','x'),values[1,:,0])
        public=reader.metadata();public['root_attributes']['header_vector'][0]=999
        public['descriptors'][0]['frames'][0]=999;public['cases'][0]['status']='failed'
        reader.components()[0]['components'].append('fake');reader.cases()[0]['status']='failed'
        reader.case_metadata('1')['frame']=42.;reader.case_summaries()[0]['status']='failed'
        again=reader.metadata()
        assert again['root_attributes']['header_vector'][0]==1 and again['cases'][0]['status']=='completed'
        assert len(reader.components()[0]['components'])==2 and np.isnan(reader.case_metadata('1')['frame'])
        assert reader.case_summaries()[0]['status']=='completed'
        assert np.isnan(again['root_attributes']['legacy_nan'])


def test_legacy_nan_metadata_discovery_never_reads_field_values(tmp_path,monkeypatch):
    path=tmp_path/'metadata_only.h5';legacy_fixture(path)
    with h5py.File(path,'r+') as h:
        h['frames/values'][:]=[np.nan,np.nan];h['fields/stress'].attrs['frames']='[NaN,NaN]'
    original=h5py.Dataset.__getitem__
    def forbid(ds,key):
        if ds.name.startswith('/fields/'):raise AssertionError('metadata copied field values')
        return original(ds,key)
    monkeypatch.setattr(h5py.Dataset,'__getitem__',forbid)
    with open_results(path) as reader:
        for fn in (reader.metadata,reader.metadata_summary,reader.cases,reader.case_summaries,reader.components):fn()
        reader.case_metadata('1');assert reader.diagnostics['reads']==0


def test_independent_ambiguous_nan_labels_remain_quantity_scoped(tmp_path):
    path=tmp_path/'scoped_nan.h5';legacy_fixture(path)
    with h5py.File(path,'r+') as h:
        h['frames/values'][:]=[np.nan,np.nan];h['fields/stress'].attrs['frames']='[NaN,NaN]'
        g=h['fields'].create_group('other');g.attrs.update(location='global',components='["scalar"]',frames='[NaN]')
        g.create_dataset('values',data=[[77.]])
    with open_results(path) as reader:
        d=reader.components('other')[0];cid=d['case_ids'][0]
        assert d['frame_axis']=='quantity' and cid=='field:other:0'
        assert np.isnan(reader.case_metadata(cid)['frame']) and len(reader.cases())==2
        np.testing.assert_array_equal(reader.read_field(cid,'other','scalar'),[77.])
        with pytest.raises(UnavailableFieldError):reader.read_field('0','other','scalar')
        reader.metadata();reader.metadata_summary()


def test_legacy_nan_partial_status_unknown_and_closed_metadata_access(tmp_path):
    from anypost import UnknownCaseError,ClosedReaderError
    path=tmp_path/'partial_nan.h5';legacy_fixture(path,partial=True)
    with h5py.File(path,'r+') as h:
        h['frames/values'][:]=[np.nan,np.nan];h['fields/stress'].attrs['frames']='[NaN,NaN]'
    reader=open_results(path)
    assert reader.case_metadata('0')['status']=='partial'
    with pytest.raises(FailedCaseError):reader.read_field('0','stress','x')
    np.testing.assert_array_equal(reader.read_stored_field('0','stress','x'),[0.,2.,4.])
    with pytest.raises(UnknownCaseError):reader.case_metadata('missing')
    reader.close()
    for fn in (reader.metadata,reader.metadata_summary,reader.cases,reader.case_summaries,reader.components,lambda:reader.case_metadata('0')):
        with pytest.raises(ClosedReaderError):fn()


def test_native_metadata_still_refuses_nonfinite_json(tmp_path):
    from anypost import ValidationError,write_results
    from anypost.shared_metadata import clone
    from test_shared_metadata import capsule,record
    with pytest.raises(ValidationError):clone({'frame':np.nan})
    cap=capsule([record('c0')]);cap['cases'][0]['frame']=np.nan
    with pytest.raises(ValidationError):write_results(tmp_path/'native.h5',cap)
    assert not (tmp_path/'native.h5').exists()
