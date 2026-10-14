import numpy as np
import h5py
import pytest
from anypost import ArchiveSink,open_results,ValidationError,ClosedReaderError

def header(n=5):
    return dict(schema="anysolver.recovery_capsule",version=1,runtime="anysolver-results/1",contract="anysolver.recovery_capsule/1",model_identity="stream-model",mesh_identity="stream-mesh",model_provenance={"snapshot_content_digest":"fixture"},solver_identity={"source":"stream-test"},patterns=("p1","p2"),geometry={},responses={"sx":np.column_stack((np.arange(n,dtype=float),np.ones(n),np.arange(n,dtype=float)*2))},associations={"sx":np.arange(n)},derived={})

def case(cid="a",active=None):
    return dict(case_id=cid,status="completed",certified=True,route="stream",active_indices=[2,1] if active is None else active)

def descriptor(key="sx",availability="reconstructable"):
    return dict(key=key,quantity_id="stress",component=key,location="element",unit="Pa",basis="global",sampling="element_sample",availability=availability)

def final(n=5,count=2):
    return dict(descriptors=(descriptor(),),associations={"sx":np.arange(n)},responses={},derived={},element_constants={"12":{"material":"steel"}},stored_superseded_cases={},case_count=count)

def test_stream_direct_append_sparse_fallback_and_header_basis(tmp_path):
    path=tmp_path/"stream.h5"
    with ArchiveSink(path) as sink:
        sink.capsule_begin(header())
        sink.capsule_case(case("a"),np.array([2.,-3.]))
        sink.capsule_case(case("b",[]),np.zeros(2))
        sink.capsule_stored("sx",np.array([1]),np.array([[10.,11.,12.,13.,14.]]))
        assert not path.exists()
        sink.capsule_end(final())
        assert not path.exists()
    assert sink.diagnostics["retained_explicit_value_bytes"]==0
    assert sink.diagnostics["peak_write_block_bytes"]<=256*1024
    with open_results(path) as reader:
        np.testing.assert_array_equal(reader.read_field("a","stress","sx"),np.arange(5)-6*np.arange(5)+2)
        np.testing.assert_array_equal(reader.read_field("b","stress","sx"),np.arange(5)+10)
        assert reader.metadata()["source_metadata"]["model_provenance"]["snapshot_content_digest"]=="fixture"
        assert reader.metadata()["source_metadata"]["element_constants"]["12"]["material"]=="steel"

@pytest.mark.parametrize("failure",["interrupted","missing_end","duplicate_case","duplicate_row","changing_width","superseded","invalid_final","changing_association"])
def test_stream_failure_preserves_destination(tmp_path,failure):
    path=tmp_path/"existing.h5";path.write_bytes(b"previous destination")
    with pytest.raises((ValidationError,RuntimeError)):
        with ArchiveSink(path) as sink:
            sink.capsule_begin(header())
            sink.capsule_case(case("a"),np.ones(2))
            sink.capsule_stored("sx",np.array([0]),np.ones((1,5)))
            if failure=="interrupted":raise RuntimeError("interrupted")
            if failure=="missing_end":return
            if failure=="duplicate_case":sink.capsule_case(case("a"),np.ones(2))
            if failure=="duplicate_row":sink.capsule_stored("sx",np.array([0]),np.ones((1,5)))
            if failure=="changing_width":
                sink.capsule_case(case("b"),np.ones(2));sink.capsule_stored("sx",np.array([1]),np.ones((1,6)))
            f=final(count=1)
            if failure=="superseded":f["stored_superseded_cases"]={"sx":[0]}
            if failure=="invalid_final":f["case_count"]=2
            if failure=="changing_association":f["associations"]["sx"]=np.arange(5)+10
            sink.capsule_end(f)
    assert path.read_bytes()==b"previous destination"
    assert not list(tmp_path.glob("*.tmp"))

def test_stream_unknown_stored_component_and_bounded_block_writes(tmp_path):
    n=100000;path=tmp_path/"blocks.h5"
    h=header(n=0);h["responses"]={};h["associations"]={}
    f=final(n=n,count=3);f["descriptors"]=(descriptor("new",availability="stored"),);f["associations"]={"new":np.arange(n)}
    with ArchiveSink(path) as sink:
        sink.capsule_begin(h)
        for i in range(3):
            sink.capsule_case(case(str(i)),np.ones(2))
            sink.capsule_stored("new",np.array([i]),np.full((1,n),float(i)))
        sink.capsule_end(f)
    assert sink.diagnostics["peak_write_block_bytes"]<=256*1024
    assert sink.diagnostics["dataset_block_writes"]>3
    with open_results(path) as reader:
        np.testing.assert_array_equal(reader.read_field("2","stress","new",[99999,0,5]),[2,2,2])
    with h5py.File(path) as handle:
        assert handle["components/0/values"].chunks[0]==1

def test_stream_zero_patterns_and_empty_rows(tmp_path):
    path=tmp_path/"empty.h5";h=header(0);h["patterns"]=();h["responses"]={};h["associations"]={}
    f=final(0,1);f["descriptors"]=(descriptor(availability="stored"),)
    with ArchiveSink(path,compression=None) as sink:
        sink.capsule_begin(h);sink.capsule_case(case(active=[]),np.empty(0))
        sink.capsule_stored("sx",np.array([0]),np.empty((1,0)));sink.capsule_end(f)
    with open_results(path) as reader:assert reader.read_field("a","stress","sx").shape==(0,)

def test_stream_no_per_case_groups(tmp_path):
    path=tmp_path/"many.h5";h=header(1);h["responses"]={};h["associations"]={}
    with ArchiveSink(path) as sink:
        sink.capsule_begin(h)
        for i in range(10000):sink.capsule_case(case(str(i)),np.ones(2))
        sink.capsule_stored("sx",np.arange(10000),np.ones((10000,1)))
        f=final(1,10000);f["descriptors"]=(descriptor(availability="stored"),);sink.capsule_end(f)
    with h5py.File(path) as handle:
        names=[];handle.visit(names.append);assert len(names)<12
    with open_results(path) as reader:assert len(reader.cases())==10000


def test_actual_solver_stream_export_and_runtime(tmp_path):
    from anysolver import PatternCase,RecoveryConfig,prepare_static_response_evaluator,export_recovery_capsule_to_sink
    from test_prepared_static_evaluator import _bar,_case
    path=tmp_path/"actual-solver.h5"
    recovery=RecoveryConfig(include_stresses=True,include_reactions=True)
    cases=[PatternCase("fallback",{"p":1.},canonical_load_case=_case(3,7).load_case),PatternCase("portable",{"p":1.})]
    with prepare_static_response_evaluator(_bar(),{"p":_case(1,1).load_case},recovery_config=recovery,stress_execution="certified_library") as evaluator:
        reference=next(evaluator.iter_batches(list(cases)))
        with ArchiveSink(path) as sink:
            export_recovery_capsule_to_sink(evaluator,cases,sink=sink,model_identity="bar",mesh_identity="bar-mesh",batch_size=1)
    with open_results(path) as reader:
        assert reader.cases()[0]["route"]=="prepared_static_response/canonical_fallback"
        for ci,cid in enumerate(("fallback","portable")):
            expected=reference.case_results[ci]["element_stresses"][1]["von_mises"]
            np.testing.assert_array_equal(reader.read_field(cid,"stress","von_mises"),np.asarray(expected).reshape(-1))
        assert reader.metadata()["source_metadata"]["model_provenance"]["export_contract"]=="anysolver.recovery_capsule/2"

def test_stream_explicit_close_abort_and_exception_after_end(tmp_path):
    path=tmp_path/"lifecycle.h5";path.write_bytes(b"old")
    sink=ArchiveSink(path);sink.capsule_begin(header());sink.capsule_case(case(),np.ones(2));sink.abort()
    assert path.read_bytes()==b"old"
    with pytest.raises(ValidationError):sink.close()
    with pytest.raises(RuntimeError):
        with ArchiveSink(path) as sink:
            sink.capsule_begin(header());sink.capsule_case(case(),np.ones(2));sink.capsule_end(final(count=1))
            raise RuntimeError("application cancelled after export finalized")
    assert path.read_bytes()==b"old"
    sink=ArchiveSink(path);sink.capsule_begin(header());sink.capsule_case(case(),np.ones(2));sink.capsule_end(final(count=1));sink.close();sink.close()
    with open_results(path) as reader:assert reader.cases()[0]["case_id"]=="a"
    with pytest.raises(ClosedReaderError):sink.capsule_case(case("late"),np.ones(2))
    assert path.exists()

def test_stream_derived_associations_and_failed_case_metadata(tmp_path):
    path=tmp_path/"derived.h5";h=header()
    h["derived"]={"eq":{"operation":"equivalent_stress","inputs":{"von_mises":"sx"}}}
    f=final(count=2);f["derived"]={"eq":{"inputs":{"von_mises":"sx"},"operation":"equivalent_stress"}}
    f["descriptors"]+=(descriptor("eq"),);f["associations"]["eq"]=np.arange(5)
    with ArchiveSink(path) as sink:
        sink.capsule_begin(h);sink.capsule_case(case("good"),np.ones(2))
        sink.capsule_case(dict(case("failed"),status="failed",certified=False,diagnostics=["retained failure"]),np.zeros(2))
        sink.capsule_end(f)
    from anypost import FailedCaseError
    with open_results(path) as reader:
        assert reader.cases()[1]["diagnostics"]==["retained failure"]
        with pytest.raises(FailedCaseError):reader.read_field("failed","stress","eq")
        np.testing.assert_array_equal(reader.read_field("good","stress","eq"),np.arange(5)*3+1)
