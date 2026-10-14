import json
import numpy as np
import pytest
from anypost import ArchiveSink,ValidationError
from test_stored_correction import capsule


def record(status="failed"):
    return dict(schema="anysolver.recovery_capsule_failure",version=1,runtime="anysolver-results/1",model_identity="exact-correction",mesh_identity="explicit-source-rows",status=status,error_type="SourceRecoveryError",error_module="source.fixture",message="native source recovery failed",input_case_ids=["c0","c1","failed-input"],diagnostics={"source_code":"SOURCE_FAILED","residual":{"numeric_kind":"nonfinite_float","value":"+Infinity"}})


def begin_one(sink):
    c=capsule();header={k:v for k,v in c.items() if k not in ("cases","coefficients","stored","descriptors")}
    sink.capsule_begin(header);sink.capsule_case(c["cases"][0],c["coefficients"][0])


def test_terminal_diagnostic_survives_completed_batch_abort_and_is_unique(tmp_path):
    path=tmp_path/"result.h5";path.write_bytes(b"previous archive");paths=[]
    for attempt in range(2):
        sink=ArchiveSink(path)
        with pytest.raises(RuntimeError,match="primary source failure"):
            with sink:
                begin_one(sink);sink.abort()
                diagnostic=sink.capsule_failed(record())
                assert diagnostic==sink.diagnostic_path
                assert sink.capsule_failed(record("cancelled"))==diagnostic
                raise RuntimeError("primary source failure")
        assert path.read_bytes()==b"previous archive" and not list(tmp_path.glob(".*.tmp"))
        data=json.loads(diagnostic.read_text())
        for key,value in record().items():assert data[key]==value
        assert data["sink"]["completed_case_count"]==1
        assert data["sink"]["completed_cases"][0]["case_id"]=="c0"
        assert data["sink"]["hdf_published"] is False
        paths.append(diagnostic)
    assert paths[0]!=paths[1] and all(path.exists() for path in paths)
    assert len(list(tmp_path.glob("result.h5.*.failure.json")))==2


def test_direct_context_and_close_failures_keep_primary_and_single_artifact(tmp_path):
    path=tmp_path/"incomplete.h5";sink=ArchiveSink(path)
    error=RuntimeError("direct context cancellation")
    with pytest.raises(RuntimeError) as caught:
        with sink:begin_one(sink);raise error
    assert caught.value is error and not path.exists()
    assert len(list(tmp_path.glob("*.failure.json")))==1
    data=json.loads(sink.diagnostic_path.read_text());assert data["message"]==str(error) and data["input_case_ids_complete"] is False
    with pytest.raises(ValidationError):sink.close()
    assert len(list(tmp_path.glob("*.failure.json")))==1


def test_non_json_failure_is_reported_without_replacing_primary(tmp_path):
    sink=ArchiveSink(tmp_path/"result.h5");begin_one(sink)
    failed=record();failed["diagnostics"]["invalid"]=np.array([1.])
    with pytest.raises(ValidationError):sink.capsule_failed(failed)
    assert sink.diagnostic_path.exists() and sink.diagnostic_error
    payload=json.loads(sink.diagnostic_path.read_text())
    assert payload["sink"]["diagnostic_record_complete"] is False
    assert payload["diagnostics_unavailable"]["error_type"]=="ValidationError"
    assert not sink.path.exists()


def test_actual_export_failure_after_certified_batch_preserves_source_and_archive(tmp_path):
    from anysolver import PatternCase,RecoveryConfig,prepare_static_response_evaluator,export_recovery_capsule_to_sink,PreparedStaticError
    from test_prepared_static_evaluator import _bar,_case
    path=tmp_path/"actual-source.h5";path.write_bytes(b"previous successful archive")
    sink=ArchiveSink(path)
    cases=[PatternCase("a",{"p":1.}),PatternCase("b",{"missing":2.})]
    with prepare_static_response_evaluator(_bar(),{"p":_case(0).load_case},recovery_config=RecoveryConfig(include_stresses=True),stress_execution="certified_library") as evaluator:
        with pytest.raises(PreparedStaticError,match="unknown source response pattern") as captured:
            with sink:export_recovery_capsule_to_sink(evaluator,cases,sink=sink,model_identity="bar",mesh_identity="bar-mesh",batch_size=1)
    assert path.read_bytes()==b"previous successful archive" and not list(tmp_path.glob(".*.tmp"))
    payload=json.loads(sink.diagnostic_path.read_text())
    assert payload["schema"]=="anysolver.recovery_capsule_failure" and payload["input_case_ids"]==["a","b"]
    assert payload["message"]==str(captured.value) and payload["diagnostics"]["pattern"]=="missing"
    assert payload["sink"]["completed_case_count"]==1 and payload["sink"]["completed_cases"][0]["case_id"]=="a"
    assert payload["sink"]["completed_cases"][0]["certified"] is True
    assert len(list(tmp_path.glob("*.failure.json")))==1
