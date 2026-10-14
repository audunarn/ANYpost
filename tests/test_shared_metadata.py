import json
from pathlib import Path
import h5py
import numpy as np
import pytest
from anypost import ArchiveSink,write_results,open_results,ValidationError,WorkingSetError,ClosedReaderError,FailedCaseError
from anypost.shared_metadata import GROUP,FIELD


def ids():return [0,-7,2**100,0]+list(range(256))


def record(cid,values=None,*,failed=False):
    result=dict(case_id=cid,status="failed" if failed else "completed",certified=not failed,route="stored",input_index=int(cid[1:]),active_indices=[],diagnostics=[{"message":"original","nested":[1,2]}],solver_info={"reaction_node_ids":[99]})
    if values is not None:result[FIELD]=values
    return result


def capsule(records):
    records=list(records)
    return dict(schema="anysolver.recovery_capsule",version=1,runtime="anysolver-results/1",model_identity="pool-model",mesh_identity="pool-mesh",solver_identity={"source":"metadata synthetic fixture"},geometry={},patterns=(),coefficients=np.empty((len(records),0)),cases=records,descriptors=({"key":"s","quantity_id":"stress","component":"s","location":"element","unit":"Pa","basis":"global","sampling":"element","availability":"stored"},),associations={"s":np.array([101])},responses={},stored={"s":{"case_indices":np.arange(len(records)),"values":np.arange(len(records),dtype=float).reshape(-1,1)}},derived={})


def stream(path,records,compression="lzf"):
    cap=capsule(records)
    header={key:value for key,value in cap.items() if key not in {"coefficients","cases","descriptors","stored"}}
    with ArchiveSink(path,compression=compression) as sink:
        sink.capsule_begin(header)
        for case in cap["cases"]:sink.capsule_case(case,np.empty(0))
        sink.capsule_stored("s",np.arange(len(records)),cap["stored"]["s"]["values"])
        sink.capsule_end({key:cap[key] for key in ("descriptors","associations","responses","derived")}|{"case_count":len(records)})
    return sink


@pytest.mark.parametrize("writer",["memory","stream"])
@pytest.mark.parametrize("compression",["lzf",None])
def test_exact_shared_case_details_summaries_and_engineering_paths(tmp_path,writer,compression):
    cases=[record("c0",ids()),record("c1",ids()),record("c2",list(reversed(ids())),failed=True),record("c3")]
    path=tmp_path/"shared.h5"
    if writer=="memory":write_results(path,capsule(cases),compression=compression)
    else:stream(path,cases,compression)
    with h5py.File(path) as h:
        assert h.attrs["version"]==2 and len(h[GROUP])==2
        metadata=json.loads(h["metadata_json"][()]);assert metadata["cases"][0][FIELD] is None
        assert metadata[GROUP]["references"]=={"0":"0","1":"0","2":"1"}
        assert all(dataset.fletcher32 and dataset.chunks[0]<=256*1024 for dataset in h[GROUP].values())
    with open_results(path) as reader:
        assert list(reader.cases())==cases and reader.metadata()["cases"]==cases
        assert reader.case_metadata("c2")==cases[2]
        assert GROUP not in reader.metadata()
        detail=reader.case_metadata("c0");detail[FIELD][0]=999;detail["diagnostics"][0]["nested"][0]=999
        assert reader.case_metadata("c0")==cases[0] and reader.case_metadata("c1")==cases[1]
        reader._case_metadata._read_table=lambda *_:pytest.fail("summary/replay must not expand metadata tables")
        summaries=reader.case_summaries();assert all(FIELD not in value and "diagnostics" not in value for value in summaries)
        assert reader.metadata_summary()["cases"]==list(summaries)
        np.testing.assert_array_equal(reader.read_field("c1","stress","s"),[1.])
        with pytest.raises(FailedCaseError):reader.read_field("c2","stress","s")
        env=reader.read_envelope("stress","s");assert env.minimum.tolist()==[0.] and env.max_case_ids.tolist()==["c3"]
        with pytest.raises(ValidationError,match="unknown"):reader.case_metadata("absent")
    with pytest.raises(ClosedReaderError):reader.case_metadata("c0")


def test_small_and_nested_ids_stay_native1_and_markerlike_data_is_literal(tmp_path):
    cases=[record("c0",[2,1,2]),record("c1")];cases[0]["arbitrary"]={"codec":"reaction_node_ids/1","reference":"0"}
    path=write_results(tmp_path/"plain.h5",capsule(cases))
    with h5py.File(path) as h:assert h.attrs["version"]==1 and GROUP not in h
    with open_results(path,working_bytes=512) as reader:assert list(reader.cases())==cases and reader.case_metadata("c0")==cases[0]


def test_explicit_detail_work_is_bounded_but_summary_admission_is_not_expansion(tmp_path):
    cases=[record(f"c{i}",ids()) for i in range(20)]
    path=write_results(tmp_path/"bounded.h5",capsule(cases))
    with open_results(path,working_bytes=400000) as reader:
        assert reader.case_metadata("c0")==cases[0]
        with pytest.raises(WorkingSetError):reader.cases()
        with pytest.raises(WorkingSetError):reader.metadata()
        assert len(reader.case_summaries())==20
    with open_results(path,working_bytes=4096) as reader:
        assert len(reader.case_summaries())==20
        with pytest.raises(WorkingSetError):reader.case_metadata("c0")


@pytest.mark.parametrize("corruption",["missing","unknown","path","identity","dangling","index","placeholder","inventory","digest","count","dtype","native1","missing_codec"])
def test_malformed_shared_metadata_refused_before_detail(tmp_path,corruption):
    path=write_results(tmp_path/f"{corruption}.h5",capsule([record("c0",ids())]))
    with h5py.File(path,"r+") as h:
        metadata=json.loads(h["metadata_json"][()]);pool=metadata[GROUP]
        if corruption=="missing":del metadata[GROUP]
        elif corruption=="unknown":pool["codec"]="unknown/2"
        elif corruption=="path":pool["path"]=["solver_info",FIELD]
        elif corruption=="identity":pool["model_identity"]="other"
        elif corruption=="dangling":pool["references"]["0"]="9"
        elif corruption=="index":pool["references"]={"-1":"0"}
        elif corruption=="placeholder":metadata["cases"][0][FIELD]=[]
        elif corruption=="inventory":h[GROUP].create_dataset("extra",data=np.zeros(1,dtype=np.uint8))
        elif corruption=="digest":pool["tables"]["0"]["sha256"]="0"*64
        elif corruption=="count":pool["tables"]["0"]["count"]+=1
        elif corruption=="dtype":del h[GROUP]["0"];h[GROUP].create_dataset("0",data=np.zeros(pool["tables"]["0"]["bytes"],dtype=np.float64))
        elif corruption=="native1":h.attrs["version"]=1
        elif corruption=="missing_codec":del pool["codec"]
        h["metadata_json"][()]=json.dumps(metadata,separators=(",",":"))
    with pytest.raises(ValidationError):open_results(path)


def test_real_pool_checksum_corruption_has_public_typed_cause(tmp_path):
    path=write_results(tmp_path/"checksum.h5",capsule([record("c0",ids())]))
    with h5py.File(path) as handle:offset=handle[GROUP]["0"].id.get_chunk_info_by_coord((0,)).byte_offset
    with path.open("r+b") as stream:
        stream.seek(offset+3);value=stream.read(1);stream.seek(offset+3);stream.write(bytes([value[0]^1]))
    with pytest.raises(ValidationError) as error:open_results(path)
    assert isinstance(error.value.__cause__,OSError)


def test_pooled_interruption_diagnostic_is_selfcontained_labelled_summary(tmp_path):
    path=tmp_path/"previous.h5";path.write_bytes(b"previous")
    with pytest.raises(RuntimeError,match="primary source"):
        with ArchiveSink(path) as sink:
            h=capsule([]);sink.capsule_begin({key:value for key,value in h.items() if key not in {"cases","coefficients","stored","descriptors"}})
            sink.capsule_case(record("c0",ids()),np.empty(0))
            raise RuntimeError("primary source")
    diagnostic=json.loads(sink.diagnostic_path.read_text())
    assert path.read_bytes()==b"previous" and diagnostic["message"]=="primary source"
    assert diagnostic["sink"]["completed_cases_are_summaries"] is True
    assert FIELD not in diagnostic["sink"]["completed_cases"][0]
    assert diagnostic["sink"]["completed_cases"][0]["case_id"]=="c0"
    assert not list(tmp_path.glob("*.tmp"))


def test_mutating_input_list_between_callbacks_interns_new_exact_table(tmp_path):
    values=ids();first=list(values);path=tmp_path/"mutated.h5"
    cap=capsule([])
    with ArchiveSink(path) as sink:
        sink.capsule_begin({key:value for key,value in cap.items() if key not in {"coefficients","cases","descriptors","stored"}})
        sink.capsule_case(record("c0",values),np.empty(0))
        values[0]=2**130
        sink.capsule_case(record("c1",values),np.empty(0))
        sink.capsule_stored("s",np.array([0,1]),np.zeros((2,1)))
        sink.capsule_end({key:cap[key] for key in ("descriptors","associations","responses","derived")}|{"case_count":2})
    with open_results(path) as reader:
        assert reader.case_metadata("c0")[FIELD]==first and reader.case_metadata("c1")[FIELD]==values
        assert reader.diagnostics["shared_case_tables"]==2


def test_small_native1_summary_marks_complete_and_pooled_always_summary(tmp_path):
    plain=write_results(tmp_path/"plain.h5",capsule([record("c0",[1,2])]))
    pooled=write_results(tmp_path/"pooled.h5",capsule([record("c0",ids())]))
    with open_results(plain) as reader:
        summary=reader.metadata_summary();assert summary["cases_complete"] is True
        assert summary["cases"]==list(reader.cases()) and FIELD not in reader.case_summaries()[0]
    with open_results(pooled) as reader:assert reader.metadata_summary()["cases_complete"] is False


@pytest.mark.parametrize("payload",[b"[true]",b"[1.0]",b"[01]",b"[-0]",b"[1,]",b"[1] "])
def test_canonical_digest_does_not_admit_wrong_integer_table(payload,tmp_path):
    import hashlib
    path=write_results(tmp_path/"invalid-table.h5",capsule([record("c0",ids())]))
    with h5py.File(path,"r+") as h:
        meta=json.loads(h["metadata_json"][()]);spec=meta[GROUP]["tables"]["0"]
        spec.update(bytes=len(payload),count=1,sha256=hashlib.sha256(payload).hexdigest())
        del h[GROUP]["0"];h[GROUP].create_dataset("0",data=np.frombuffer(payload,dtype=np.uint8))
        h["metadata_json"][()]=json.dumps(meta)
    with pytest.raises(ValidationError):open_results(path)


def test_duplicate_native2_reference_keys_refused(tmp_path):
    path=write_results(tmp_path/"duplicate-json.h5",capsule([record("c0",ids())]))
    with h5py.File(path,"r+") as h:
        text=h["metadata_json"][()].decode()
        text=text.replace('"references":{"0":"0"}','"references":{"0":"9","0":"0"}')
        h["metadata_json"][()]=text
    with pytest.raises(ValidationError,match="duplicate"):open_results(path)
