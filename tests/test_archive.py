import numpy as np
import pytest
from anypost import write_results, open_results

@pytest.fixture
def capsule():
    n=5
    return dict(schema="anysolver.recovery_capsule",version=1,runtime="anysolver-results/1",
                model_identity="m1",mesh_identity="mesh1",solver_identity={"source":"fixture"},
                geometry={"positions":np.zeros((n,3)),"node_ids":np.arange(n)},patterns=("p",),
                coefficients=np.array([[1.],[2.],[0.]]),
                cases=({"case_id":"a","status":"completed","certified":True,"route":"stored"},
                       {"case_id":"b","status":"completed","certified":True,"route":"stored"},
                       {"case_id":"failed","status":"failed","certified":False,"route":"failed","diagnostics":["failure retained"]}),
                descriptors=({"key":"sx","quantity_id":"stress","component":"x","location":"node","unit":"Pa","basis":"global","sampling":"node","availability":"stored"},),
                associations={"sx":np.arange(n)},responses={},derived={},
                stored={"sx":{"case_indices":np.array([0,1,2]),"values":np.array([[1.,2.,3.,4.,5.],[1.,1.,4.,2.,7.],[99.,99.,99.,99.,99.]])}})

def test_roundtrip(tmp_path,capsule):
    path=write_results(tmp_path/"sample.h5",capsule)
    with open_results(path) as reader:
        assert reader.components()[0]["unit"]=="Pa"
        assert reader.diagnostics["reads"]==0
        assert reader.cases()[2]["diagnostics"]==["failure retained"]
        assert reader.diagnostics["reads"]==0
        values=reader.read_field("a","stress","x",[4,0,4,2])
        np.testing.assert_array_equal(values,[5,1,5,3])
        assert not values.flags.writeable
        env=reader.read_envelope("stress","x")
        np.testing.assert_array_equal(env.minimum,[1,1,3,2,5])
        np.testing.assert_array_equal(env.maximum,[1,2,4,4,7])
        assert env.min_case_ids[0]==env.max_case_ids[0]=="a"
        np.testing.assert_array_equal(reader.associations("stress","x"),np.arange(5))
        assert reader.geometry()["positions"].shape==(5,3)
from anypost import (ValidationError,IdentityError,FailedCaseError,UnavailableFieldError,
                     WorkingSetError,RuntimeDependencyError,ClosedReaderError)
import copy
import h5py
import sys
from types import SimpleNamespace

def test_failed_identity_closed_and_working(tmp_path,capsule):
    path=write_results(tmp_path/"refusal.h5",capsule)
    with pytest.raises(IdentityError): open_results(path,expected_model_identity="other")
    with open_results(path,working_bytes=32) as reader:
        with pytest.raises(WorkingSetError): reader.read_field("a","stress","x")
        with pytest.raises(FailedCaseError): reader.read_field("failed","stress","x")
        with pytest.raises(FailedCaseError): reader.read_envelope("stress","x",case_ids=["failed"])
    with pytest.raises(ClosedReaderError): reader.read_field("a","stress","x")

@pytest.mark.parametrize("mutation",["certification","coefficient_shape","response_shape","unknown_derived","cycle","stored_rows","nonfinite","dtype","association_shape"])
def test_validation_preserves_existing_destination(tmp_path,capsule,mutation):
    path=write_results(tmp_path/"atomic.h5",capsule)
    previous=path.read_bytes()
    c=copy.deepcopy(capsule)
    if mutation=="certification": c["cases"][2]["certified"]=True
    elif mutation=="coefficient_shape": c["coefficients"]=np.zeros((2,1))
    elif mutation=="response_shape": c["responses"]={"sx":np.zeros((5,1))}
    elif mutation=="unknown_derived": c["derived"]={"sx":{"operation":"x","inputs":{"x":"absent"}}}
    elif mutation=="cycle": c["derived"]={"sx":{"operation":"x","inputs":{"x":"sx"}}}
    elif mutation=="stored_rows": c["stored"]["sx"]["case_indices"]=np.array([0,0,1])
    elif mutation=="nonfinite": c["stored"]["sx"]["values"][0,0]=np.nan
    elif mutation=="dtype": c["associations"]["sx"]=np.arange(5,dtype=float)
    elif mutation=="association_shape": c["associations"]["sx"]=np.zeros((5,1,1),dtype=int)
    with pytest.raises(ValidationError): write_results(path,c)
    assert path.read_bytes()==previous
    assert not list(tmp_path.glob("*.tmp"))

def test_cache_external_and_eviction(tmp_path,capsule):
    path=write_results(tmp_path/"cache.h5",capsule)
    with open_results(path,cache_bytes=64) as reader:
        a=reader.read_field("a","stress","x")
        assert reader.read_field("a","stress","x") is a
        assert reader.diagnostics["cache_hits"]==1
        reader.set_external_cache_bytes("viewer",40)
        assert reader.diagnostics["cache_resident_bytes"]<=64
        assert reader.diagnostics["external_cache_bytes"]==40
        with pytest.raises(WorkingSetError): reader.set_external_cache_bytes("other",30)
        reader.set_external_cache_bytes("viewer",0)
        reader.read_field("b","stress","x")
        assert reader.diagnostics["cache_peak_bytes"]<=64

def test_runtime_delegation_override_and_identity(tmp_path,capsule,monkeypatch):
    c=copy.deepcopy(capsule)
    c["descriptors"][0]["availability"]="reconstructable"
    c["responses"]={"sx":np.column_stack((np.arange(5,dtype=float)+10,np.arange(5,dtype=float)*2))}
    c["stored"]["sx"]["case_indices"]=np.array([1])
    c["stored"]["sx"]["values"]=np.array([[90,91,92,93,94]],dtype=float)
    c["cases"][0]["active_indices"]=[1]
    path=write_results(tmp_path/"replay.h5",c)
    monkeypatch.setitem(sys.modules,"anysolver_results",None)
    with open_results(path) as reader:
        np.testing.assert_array_equal(reader.read_field("b","stress","x"),[90,91,92,93,94])
        with pytest.raises(RuntimeDependencyError): reader.read_field("a","stress","x")
    monkeypatch.setitem(sys.modules,"anysolver_results",SimpleNamespace(RUNTIME_ID="wrong"))
    with open_results(path) as reader:
        with pytest.raises(RuntimeDependencyError): reader.read_field("a","stress","x")
    calls=[]
    def reconstruct(responses,coefficients,*,active_indices=None):
        calls.append(active_indices)
        out=responses[:,0].copy()
        columns = active_indices if active_indices is not None else range(1,len(coefficients)+1)
        for column, coefficient in zip(columns, coefficients):
            out+=responses[:,column]*coefficient
        return out
    monkeypatch.setitem(sys.modules,"anysolver_results",SimpleNamespace(RUNTIME_ID="anysolver-results/1",reconstruct_response=reconstruct))
    with open_results(path,working_bytes=300) as reader:
        np.testing.assert_array_equal(reader.read_field("a","stress","x",slice(None,None,-1)),[22,19,16,13,10])
        assert calls and all(item==[1] for item in calls)
        assert reader.diagnostics["working_peak_bytes"]<=300

def test_derived_delegation(tmp_path,capsule,monkeypatch):
    c=copy.deepcopy(capsule)
    desc=dict(c["descriptors"][0],key="eq",quantity_id="equivalent",availability="reconstructable")
    c["descriptors"]+=(desc,)
    c["associations"]["eq"]=c["associations"]["sx"].copy()
    c["derived"]={"eq":{"operation":"fixture","inputs":{"canonical":"sx"}}}
    calls=[]
    def derive(specification,inputs):
        calls.append(specification["operation"])
        return inputs["canonical"]*2
    monkeypatch.setitem(sys.modules,"anysolver_results",SimpleNamespace(RUNTIME_ID="anysolver-results/1",derive_field=derive))
    path=write_results(tmp_path/"derived.h5",c)
    with open_results(path) as reader:
        np.testing.assert_array_equal(reader.read_field("a","equivalent","x"),[2,4,6,8,10])
        assert calls==["fixture"]
    c["associations"]["eq"][0]=99
    with pytest.raises(ValidationError): write_results(path,c)

def test_empty_and_unavailable(tmp_path,capsule):
    c=copy.deepcopy(capsule)
    c["associations"]["sx"]=np.array([],dtype=int)
    c["stored"]["sx"]["values"]=np.zeros((3,0))
    c["geometry"]={}
    path=write_results(tmp_path/"empty.h5",c,compression=None)
    with open_results(path) as reader:
        assert reader.read_field("a","stress","x").shape==(0,)
        assert reader.read_envelope("stress","x").minimum.shape==(0,)
    c["descriptors"][0]["availability"]="unavailable"
    c["stored"]={}
    path=write_results(path,c)
    with open_results(path) as reader:
        assert reader.components()[0]["availability"]=="unavailable"
        with pytest.raises(UnavailableFieldError): reader.read_field("a","stress","x")

def test_malformed_archive_metadata(tmp_path,capsule):
    path=write_results(tmp_path/"bad.h5",capsule)
    with h5py.File(path,"r+") as h: h.attrs["complete"]=False
    with pytest.raises(ValidationError): open_results(path)


def test_dataset_count_independent_of_case_count(tmp_path,capsule):
    c=copy.deepcopy(capsule); n=10000
    c["cases"]=tuple(dict(case_id=str(i),status="completed",certified=True,route="stored") for i in range(n))
    c["coefficients"]=np.zeros((n,1))
    c["stored"]["sx"]={"case_indices":np.arange(n),"values":np.zeros((n,5))}
    path=write_results(tmp_path/"many.h5",c)
    with h5py.File(path) as h:
        names=[]; h.visit(names.append)
        assert len(names)<15
    with open_results(path) as reader:
        assert len(reader.cases())==10000
        np.testing.assert_array_equal(reader.read_field("9999","stress","x"),np.zeros(5))

def test_missing_stored_row_needs_no_runtime(tmp_path,capsule,monkeypatch):
    c=copy.deepcopy(capsule)
    c["stored"]["sx"]["case_indices"]=np.array([0])
    c["stored"]["sx"]["values"]=c["stored"]["sx"]["values"][:1]
    path=write_results(tmp_path/"missing.h5",c)
    monkeypatch.setitem(sys.modules,"anysolver_results",None)
    with open_results(path) as reader:
        with pytest.raises(UnavailableFieldError):reader.read_field("b","stress","x")
        with pytest.raises(UnavailableFieldError):reader.read_field("b","stress","x",[])

def test_explicit_row_corruption_refused(tmp_path,capsule):
    path=write_results(tmp_path/"indices.h5",capsule)
    with h5py.File(path,"r+") as h:h["components/0/case_indices"][0]=2
    with pytest.raises(ValidationError):open_results(path)

def test_cache_clear_preserves_live_external(tmp_path,capsule):
    path=write_results(tmp_path/"clear.h5",capsule)
    with open_results(path,cache_bytes=256) as reader:
        reader.read_field("a","stress","x")
        reader.set_external_cache_bytes("viewer",40)
        reader.clear_cache()
        assert reader.diagnostics["cache_resident_bytes"]==40
        assert reader.diagnostics["external_cache_bytes"]==40

def test_authoritative_last_tie_policy_and_source_metadata(tmp_path,capsule):
    c=copy.deepcopy(capsule); c["descriptors"][0]["tie_policy"]="last"
    c["exporter_provenance"]={"source_revision":"fixture-v1"}
    path=write_results(tmp_path/"last.h5",c)
    with open_results(path) as reader:
        assert reader.read_envelope("stress","x").min_case_ids[0]=="b"
        assert reader.metadata()["source_metadata"]["exporter_provenance"]["source_revision"]=="fixture-v1"

def test_working_cap_requires_spatial_selection(tmp_path,capsule):
    c=copy.deepcopy(capsule); n=1000
    c["geometry"]={}
    c["associations"]["sx"]=np.arange(n)
    c["stored"]["sx"]["values"]=np.ones((3,n))
    path=write_results(tmp_path/"large.h5",c)
    with open_results(path,working_bytes=4096) as reader:
        with pytest.raises(WorkingSetError):reader.read_field("a","stress","x")
        values=reader.read_field("a","stress","x",slice(10,20))
        np.testing.assert_array_equal(values,np.ones(10))
        assert reader.diagnostics["working_peak_bytes"]<=4096

def test_external_link_refused(tmp_path,capsule):
    path=write_results(tmp_path/"link.h5",capsule)
    with h5py.File(path,"r+") as h:h["external"]=h5py.ExternalLink("another.h5","/field")
    with pytest.raises(ValidationError):open_results(path)

def test_original_sparse_column_order_and_empty_active(tmp_path,capsule,monkeypatch):
    c=copy.deepcopy(capsule)
    c["patterns"]=("left","middle","right")
    c["coefficients"]=np.array([[0.,-3.,2.],[1.,2.,3.],[0.,0.,0.]])
    c["cases"][0]["active_indices"]=[3,2]
    c["cases"][1]["active_indices"]=[]
    c["descriptors"][0]["availability"]="reconstructable"
    c["responses"]={"sx":np.array([[1e16,1.,1e16,-1e16],[2.,3.,4.,5.],[10.,20.,30.,40.],[11.,21.,31.,41.],[12.,22.,32.,42.]])}
    c["stored"]={}
    captured=[]
    def reconstruct(responses,coefficients,*,active_indices=None):
        captured.append((list(active_indices),coefficients.copy()))
        columns=np.asarray(active_indices,dtype=int)
        # The actual solver admitted packet accumulation is active-column matmul.
        return responses[:,0]+responses[:,columns]@coefficients
    monkeypatch.setitem(sys.modules,"anysolver_results",SimpleNamespace(RUNTIME_ID="anysolver-results/1",reconstruct_response=reconstruct))
    path=write_results(tmp_path/"sparse.h5",c)
    with open_results(path) as reader:
        expected=c["responses"]["sx"][:,0]+c["responses"]["sx"][:,[3,2]]@np.array([2.,-3.])
        np.testing.assert_array_equal(reader.read_field("a","stress","x"),expected)
        np.testing.assert_array_equal(reader.read_field("b","stress","x"),c["responses"]["sx"][:,0])
        assert captured[0][0]==[3,2]
        np.testing.assert_array_equal(captured[0][1],[2.,-3.])
        assert captured[1][0]==[] and captured[1][1].size==0
    c["cases"][0]["active_indices"]=[0]
    with pytest.raises(ValidationError):write_results(path,c)

def test_geometry_topology_masks_and_invalid_indices(tmp_path,capsule):
    c=copy.deepcopy(capsule)
    c["geometry"].update(triangles=np.array([[0,1,2]]),lines=np.array([[2,3]]),triangle_to_element=np.array([21]),element_ids=np.array([21]),active_elements=np.array([True]))
    path=write_results(tmp_path/"geometry.h5",c)
    with open_results(path) as reader:
        assert reader.geometry()["active_elements"].dtype==np.dtype(bool)
        np.testing.assert_array_equal(reader.geometry()["triangle_to_element"],[21])
    c["geometry"]["triangles"][0,0]=100
    with pytest.raises(ValidationError):write_results(path,c)

def test_coefficient_row_cached_across_chunks_components_and_cases(tmp_path,capsule,monkeypatch):
    c=copy.deepcopy(capsule);n=20
    c["geometry"]={}; c["patterns"]=("p1","p2")
    c["coefficients"]=np.array([[2.,-3.],[5.,7.],[0.,0.]])
    c["cases"][0]["active_indices"]=[2,1]
    c["descriptors"][0]["availability"]="reconstructable"
    c["descriptors"]+=(dict(c["descriptors"][0],key="sy",component="y"),)
    c["associations"]={k:np.arange(n) for k in ("sx","sy")}
    basis=np.column_stack((np.arange(n,dtype=float),np.ones(n),np.arange(n,dtype=float)*3))
    c["responses"]={"sx":basis,"sy":basis*2};c["stored"]={}
    def reconstruct(responses,coefficients,*,active_indices=None):
        columns=np.arange(1,len(coefficients)+1) if active_indices is None else np.array(active_indices)
        return responses[:,0]+responses[:,columns]@coefficients
    monkeypatch.setitem(sys.modules,"anysolver_results",SimpleNamespace(RUNTIME_ID="anysolver-results/1",reconstruct_response=reconstruct))
    path=write_results(tmp_path/"coef-cache.h5",c)
    with open_results(path,cache_bytes=8192,working_bytes=512) as reader:
        expected=basis[:,0]+basis[:,[2,1]]@np.array([-3.,2.])
        np.testing.assert_array_equal(reader.read_field("a","stress","x"),expected)
        assert reader.diagnostics["coefficient_reads"]==1
        assert reader.diagnostics["coefficient_hits"]>=1
        np.testing.assert_array_equal(reader.read_field("a","stress","y"),expected*2)
        assert reader.diagnostics["coefficient_reads"]==1
        np.testing.assert_array_equal(reader.read_field("b","stress","x"),basis[:,0]+basis[:,1:]@c["coefficients"][1])
        assert reader.diagnostics["coefficient_reads"]==2
        assert reader.diagnostics["cache_peak_bytes"]<=8192
        reader.clear_cache()
        reader.read_field("a","stress","x")
        assert reader.diagnostics["coefficient_reads"]==3

def test_stored_paths_do_not_read_coefficients(tmp_path,capsule):
    path=write_results(tmp_path/"stored-coefficients.h5",capsule)
    with open_results(path) as reader:
        reader.read_field("a","stress","x")
        reader.read_envelope("stress","x")
        assert reader.diagnostics["coefficient_reads"]==0
