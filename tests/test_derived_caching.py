import json
import h5py
import numpy as np
import pytest
from anypost import write_results,open_results,ValidationError,WorkingSetError
import anysolver_results as runtime
from anysolver_results.formulas import SHELL_LINEAR_COMPONENTS


def shell_capsule(n=129,*,binary=True):
    rng=np.random.default_rng(731)
    coefficients=np.array([[.5,-2.,3.],[-1.,4.,-.25]])
    associations=np.column_stack((np.arange(n)//2+100,np.arange(n)%2))
    responses={key:(rng.integers(-8,9,size=(n,4)).astype(float) if binary else rng.normal(size=(n,4))) for key in SHELL_LINEAR_COMPONENTS}
    descriptors=[dict(key=key,quantity_id="stress",component=key,location="element",unit="Pa",basis="compiled",sampling="station",availability="reconstructable") for key in SHELL_LINEAR_COMPONENTS]
    inputs={key:key for key in SHELL_LINEAR_COMPONENTS}
    derived={"vm":{"operation":"shell_von_mises","inputs":inputs},"principal":{"operation":"shell_principal_stress","surface":"top","rank":1,"inputs":inputs}}
    for key in derived:descriptors.append(dict(descriptors[0],key=key,component=key))
    return dict(schema="anysolver.recovery_capsule",version=1,runtime="anysolver-results/1",model_identity="shell-fixture",mesh_identity="stations",solver_identity={},geometry={},patterns=("p1","p2","p3"),coefficients=coefficients,cases=tuple(dict(case_id=cid,status="converged",certified=True,route="fixture",active_indices=[3,1,2]) for cid in ("a","b")),descriptors=tuple(descriptors),associations={d["key"]:associations.copy() for d in descriptors},responses=responses,stored={},derived=derived)


def reference(capsule,key,case=0,selection=None,group_size=None):
    indices=np.arange(len(capsule["associations"][key])) if selection is None else np.arange(len(capsule["associations"][key]))[selection]
    indices=np.atleast_1d(indices)
    active=capsule["cases"][case]["active_indices"]
    weights=capsule["coefficients"][case][np.asarray(active,dtype=np.int64)-1]
    spec=capsule["derived"][key]
    def block(chosen):
        inputs={name:runtime.reconstruct_response(capsule["responses"][dep][chosen],weights,active_indices=active) for name,dep in spec["inputs"].items()}
        return runtime.derive_field(spec,inputs)
    if group_size is None:return block(indices)
    return np.concatenate([block(indices[start:start+group_size]) for start in range(0,len(indices),group_size)])


@pytest.mark.parametrize("key",["vm","principal"])
def test_derived_runtime_exact_binary_parity_across_tiles_sparse_and_selection(tmp_path,key):
    c=shell_capsule(600);path=write_results(tmp_path/"parity.h5",c)
    expected=reference(c,key)
    np.testing.assert_array_equal(reference(c,key,group_size=1),expected)
    np.testing.assert_array_equal(reference(c,key,group_size=17),expected)
    with open_results(path,working_bytes=65536) as reader:
        np.testing.assert_array_equal(reader.read_field("a","stress",key),expected)
        np.testing.assert_array_equal(reader.read_field("a","stress",key,slice(None,None,-1)),expected[::-1])
        chosen=np.array([599,0,599,27,2,3])
        np.testing.assert_array_equal(reader.read_field("a","stress",key,chosen),expected[chosen])
        assert reader.diagnostics["working_peak_bytes"]<=65536
        assert 0<reader.diagnostics["largest_compute_tile"]<600
        with pytest.raises(WorkingSetError):reader.read_field("b","stress",key,np.zeros(5000,dtype=np.int64))


def test_real_float_runtime_parity_uses_existing_solver_tolerances(tmp_path):
    c=shell_capsule(257,binary=False);path=write_results(tmp_path/"floating.h5",c)
    with open_results(path,working_bytes=65536) as reader:
        for key in ("vm","principal"):
            expected=reference(c,key)
            # Established runtime/solver development comparison, unchanged.
            np.testing.assert_allclose(reference(c,key,group_size=1),expected,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(reader.read_field("a","stress",key),expected,rtol=1e-12,atol=1e-12)


def test_association_proofs_cache_across_cases_siblings_and_eviction(tmp_path):
    c=shell_capsule(129);path=write_results(tmp_path/"proof.h5",c)
    with open_results(path,cache_bytes=2048) as reader:
        np.testing.assert_array_equal(reader.read_field("a","stress","vm"),reference(c,"vm"))
        count=reader.diagnostics["association_reads"]
        assert count>0 and reader.diagnostics["association_verifications"]==8
        reader.set_external_cache_bytes("evict",2048)
        reader.clear_cache()
        np.testing.assert_array_equal(reader.read_field("b","stress","vm"),reference(c,"vm",case=1))
        assert reader.diagnostics["association_reads"]==count
        assert reader.diagnostics["cache_resident_bytes"]<=2048
        reader.set_external_cache_bytes("evict",0)
        np.testing.assert_array_equal(reader.read_field("b","stress","principal"),reference(c,"principal",case=1))
        assert reader.diagnostics["association_verifications"]==16
        assert reader.diagnostics["cache_peak_bytes"]<=2048


def test_wrong_association_refused_before_derivation_even_for_other_selection(tmp_path):
    c=shell_capsule(129);path=write_results(tmp_path/"wrong.h5",c)
    with h5py.File(path,"r+") as handle:
        meta=json.loads(handle["metadata_json"][()].decode())
        group=meta["component_groups"]["shear_xz"]
        parent=handle[f"components/{group}"]
        # The writer legitimately hard-links identical association tables.
        # Detach only the dependency under test so the malformed edge differs.
        original=parent['associations'][()]
        del parent['associations']
        parent.create_dataset('associations',data=original,chunks=True,fletcher32=True)
        parent['associations'][100,0]=9999
        assert parent['associations'][100,0]!=handle[f"components/{meta['component_groups']['vm']}/associations"][100,0]
    with open_results(path) as reader:
        with pytest.raises(ValidationError,match="associations differ"):
            reader.read_field("a","stress","vm",[0])
        assert reader.diagnostics["reconstructions"]==0


@pytest.mark.parametrize("storage",["virtual","raw_external"])
def test_native_external_storage_refused_before_successful_fill_zero(tmp_path,storage):
    c=shell_capsule(129);c["descriptors"][0]["availability"]="stored"
    key=c["descriptors"][0]["key"]
    c["responses"].pop(key)
    c["stored"][key]={"case_indices":np.array([0,1]),"values":np.ones((2,129))}
    path=write_results(tmp_path/"external.h5",c)
    with h5py.File(path,"r+") as handle:
        meta=json.loads(handle["metadata_json"][()].decode());group=handle[f"components/{meta['component_groups'][key]}"]
        del group["values"]
        if storage=="virtual":
            layout=h5py.VirtualLayout(shape=(2,129),dtype=np.float64)
            layout[:]=h5py.VirtualSource(str(tmp_path/"absent-source.h5"),"values",shape=(2,129))
            group.create_virtual_dataset("values",layout,fillvalue=0.)
        else:
            group.create_dataset("values",shape=(2,129),dtype=np.float64,external=[(str(tmp_path/"absent-values.raw"),0,h5py.h5f.UNLIMITED)])
    with pytest.raises(ValidationError,match="self-contained"):
        open_results(path)


def test_sibling_compute_grids_share_basis_and_public_association_cache(tmp_path):
    c=shell_capsule(5000);path=write_results(tmp_path/"grid.h5",c)
    with open_results(path) as reader:
        reader.read_field("a","stress","vm")
        old_reads=reader.diagnostics["reads"]
        reader.read_field("a","stress","principal")
        # The sibling uses the same primitive tile boundaries: only its own
        # previously-unverified association blocks need native field I/O.
        assert reader.diagnostics["reads"]-old_reads<20
        assoc=reader.associations("stress","vm")
        assert reader.associations("stress","vm") is assoc
        assert not assoc.flags.writeable
        assert reader.diagnostics["largest_compute_tile"]>256*1024//reader._cost("vm")


def test_additive_prepared_runtime_abi_cache_order_and_working_budget(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import sys
    c=shell_capsule(600);path=write_results(tmp_path/"prepared.h5",c)
    calls=[]
    def prepare_response(responses,*,active_indices=None,combination='legacy_blas/1'):
        active=tuple(range(1,responses.shape[1])) if active_indices is None else tuple(active_indices)
        calls.append(active)
        return runtime.prepare_response(responses,active_indices=active_indices,combination=combination)
    def reconstruct_prepared_response(prepared,packed_coefficients,*,combination=None):
        return runtime.reconstruct_prepared_response(prepared,packed_coefficients,combination=combination)
    fake=SimpleNamespace(RUNTIME_ID=runtime.RUNTIME_ID,
        SUPPORTED_RUNTIME_IDS=runtime.SUPPORTED_RUNTIME_IDS,
        SUPPORTED_COMBINATIONS=runtime.SUPPORTED_COMBINATIONS,
        prepare_response=prepare_response,reconstruct_prepared_response=reconstruct_prepared_response,
        derive_field=runtime.derive_field,reconstruct_response=runtime.reconstruct_response)
    monkeypatch.setitem(sys.modules,"anysolver_results",fake)
    with open_results(path,working_bytes=65536) as reader:
        np.testing.assert_array_equal(reader.read_field("a","stress","principal"),reference(c,"principal"))
        creations=reader.diagnostics["prepared_response_creations"]
        np.testing.assert_array_equal(reader.read_field("b","stress","principal"),reference(c,"principal",case=1))
        assert reader.diagnostics["prepared_response_creations"]==creations
        assert reader.diagnostics["prepared_response_hits"]>=creations
        assert all(active==(3,1,2) for active in calls)
        assert reader.diagnostics["working_peak_bytes"]<=65536
        assert reader.diagnostics["cache_resident_bytes"]<=reader.diagnostics["cache_bytes"]
        # A declared empty active sequence still prepares baseline once.
    c["cases"][0]["active_indices"]=[]
    path=write_results(tmp_path/"empty-active-prepared.h5",c)
    with open_results(path) as reader:
        np.testing.assert_array_equal(reader.read_field("a","stress","vm"),reference(c,"vm"))
    assert () in calls


def test_prepared_runtime_incomplete_or_bad_accounting_refused(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import sys
    c=shell_capsule(20);path=write_results(tmp_path/"bad-prepared.h5",c)
    from anypost import RuntimeDependencyError
    declared=dict(RUNTIME_ID=runtime.RUNTIME_ID,SUPPORTED_RUNTIME_IDS=runtime.SUPPORTED_RUNTIME_IDS,SUPPORTED_COMBINATIONS=runtime.SUPPORTED_COMBINATIONS)
    for fake,error,message in (
        (SimpleNamespace(**declared,prepare_response=lambda *a,**kw:None),RuntimeDependencyError,'incomplete prepared response ABI'),
        (SimpleNamespace(**declared,prepare_response=lambda *a,**kw:SimpleNamespace(nbytes=0),reconstruct_prepared_response=lambda *a,**kw:np.zeros(20),derive_field=runtime.derive_field),ValidationError,'account exact')):
        monkeypatch.setitem(sys.modules,"anysolver_results",fake)
        with open_results(path) as reader:
            with pytest.raises(error,match=message):
                reader.read_field("a","stress","vm")


def test_fortran_source_layout_retains_exact_runtime_values(tmp_path):
    c=shell_capsule(10000);path=write_results(tmp_path/"layout.h5",c)
    with open_results(path) as reader:
        values=reader.read_field("a","stress","principal")
        np.testing.assert_array_equal(values,reference(c,"principal"))
        # Source columns stay immutable float64; layout is a storage decision.
        matrices=[item[0] for key,item in reader._cache._values.items() if key[0]=="basis"]
        assert matrices and all(a.flags.f_contiguous and not a.flags.writeable for a in matrices)



def test_actual_prepared_runtime_cache_bytes_and_exact_replay(tmp_path):
    assert callable(runtime.prepare_response) and callable(runtime.reconstruct_prepared_response)
    c=shell_capsule(10000);path=write_results(tmp_path/"actual-prepared.h5",c)
    with open_results(path,working_bytes=256*1024**2) as reader:
        np.testing.assert_array_equal(reader.read_field("a","stress","principal"),reference(c,"principal"))
        created=reader.diagnostics["prepared_response_creations"]
        assert created>0
        np.testing.assert_array_equal(reader.read_field("b","stress","principal"),reference(c,"principal",case=1))
        assert reader.diagnostics["prepared_response_creations"]==created
        assert reader.diagnostics["prepared_response_hits"]>=created
        prepared=[item[0] for key,item in reader._cache._values.items() if key[0]=="prepared_response"]
        assert prepared and all(type(item) is runtime.PreparedResponse for item in prepared)
        assert all(item.nbytes==item.baseline.nbytes+item.selected.nbytes for item in prepared)
        assert all(not item.baseline.flags.writeable and not item.selected.flags.writeable for item in prepared)
        assert reader.diagnostics["cache_resident_bytes"]<=reader.diagnostics["cache_bytes"]
        assert reader.diagnostics["working_peak_bytes"]<=reader.working_bytes
        reader.clear_cache()
        reader.read_field("a","stress","vm")
        assert reader.diagnostics["prepared_response_creations"]>created
