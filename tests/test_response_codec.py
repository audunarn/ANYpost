import json
import h5py
import numpy as np
import pytest
from anypost import ArchiveSink, write_results, open_results, ValidationError
from test_stream import header, case, descriptor, final
from test_checksum_corruption import corrupt_payload, assert_checksum_error
from test_derived_caching import shell_capsule, reference


def native(n=35):
    c=shell_capsule(n)
    associations=np.column_stack((np.repeat(np.arange(n//7)*13+100,7),np.tile(np.arange(7),n//7)))
    for key in c["associations"]:c["associations"][key]=associations.copy()
    return c


@pytest.mark.parametrize("compression",[None,"lzf"])
def test_codec_exact_flat_decode_sparse_baseline_and_override(tmp_path,compression):
    c=native(); key=next(iter(c["responses"]))
    c["stored"]={key:{"case_indices":np.array([1]),"values":np.arange(35,dtype=float)[None,:]}}
    c["cases"]=tuple(dict(case,active_indices=[] if i==1 else [3,1,2]) for i,case in enumerate(c["cases"]))
    path=write_results(tmp_path/"encoded.h5",c,compression=compression)
    with h5py.File(path,"r") as h:
        meta=json.loads(h["metadata_json"][()].decode());ds=h[f"components/{meta['component_groups'][key]}/responses"]
        assert ds.shape==(5,7,4) and ds.chunks[1:]==(1,4)
        np.testing.assert_array_equal(ds[:].reshape(35,4),c["responses"][key])
        assert ds.fletcher32 and ds.shuffle
    for indices in (None,np.arange(0,35,7),np.array([34,0,34,13,9,0]),slice(None,None,-1),np.array([],dtype=int)):
        with open_results(path,cache_bytes=512,working_bytes=32768) as reader:
            np.testing.assert_array_equal(reader.read_field("a","stress","principal",indices),reference(c,"principal",selection=indices))
            chosen=np.arange(35) if indices is None else np.arange(35)[indices]
            np.testing.assert_array_equal(reader.read_field("b","stress",key,indices),np.arange(35,dtype=float)[chosen])
            other=list(c["responses"])[1]
            np.testing.assert_array_equal(reader.read_field("b","stress",other,indices),c["responses"][other][chosen,0])
            reader.clear_cache()
            np.testing.assert_array_equal(reader.read_field("a","stress","principal",indices),reference(c,"principal",selection=indices))
            assert reader.diagnostics["cache_peak_bytes"]<=512 and reader.diagnostics["working_peak_bytes"]<=32768


@pytest.mark.parametrize("ids",[[1,2,3,4],[1,1,2],[1,1,2,2,1,1],[1,2,1,2],[]])
def test_nonuniform_or_noncontiguous_ids_keep_flat_codec(tmp_path,ids):
    c=header(len(ids));c["associations"]={"sx":np.array(ids,dtype=int)}
    c.update(cases=(case(),),coefficients=np.array([[2.,-3.]]),descriptors=(descriptor(),),stored={})
    path=write_results(tmp_path/"flat.h5",c)
    with h5py.File(path) as h:assert h["components/0/responses"].ndim==2 and "codec" not in h["components/0/responses"].attrs
    with open_results(path) as reader:assert reader.read_field("a","stress","sx").shape==(len(ids),)


def test_stream_seven_samples_early_and_final_exact_binding(tmp_path):
    h=header(35);assoc=np.column_stack((np.repeat([91,17,44,8,103],7),np.tile(np.arange(7),5)))
    h["associations"]={"sx":assoc};end=final(35,count=2);end["associations"]={"sx":assoc};end["responses"]=h["responses"]
    path=tmp_path/"stream.h5"
    with ArchiveSink(path) as sink:
        sink.capsule_begin(h)
        assert sink._group("sx")["responses"].shape==(5,7,3)
        sink.capsule_case(case(),np.array([2.,-3.]))
        sink.capsule_case(case("b",[]),np.zeros(2))
        sink.capsule_stored("sx",np.array([1]),np.arange(35,dtype=float)[None,:])
        sink.capsule_end(end)
    assert sink.diagnostics["peak_write_block_bytes"]<=256*1024
    with open_results(path) as reader:
        selected=np.array([34,0,34,7,8])
        np.testing.assert_array_equal(reader.read_field("a","stress","sx",selected),(np.arange(35)*-5+2)[selected])
        np.testing.assert_array_equal(reader.read_field("b","stress","sx",selected),np.arange(35)[selected])
        np.testing.assert_array_equal(reader.associations("stress","sx",selected),assoc[selected])
    old=path.read_bytes()
    with pytest.raises(ValidationError):
        with ArchiveSink(path) as sink:
            sink.capsule_begin(h);sink.capsule_case(case(),np.array([2.,-3.]))
            broken=final(35,1);bad=assoc.copy();bad[0,1]=999;broken["associations"]={"sx":bad};sink.capsule_end(broken)
    assert path.read_bytes()==old


@pytest.mark.parametrize("attribute,value",[("codec","unknown/1"),("samples_per_entity",0),("samples_per_entity",2),("entity_count",6),("logical_rows",34),("association_sha256","bad")])
def test_malformed_codec_attributes_typed_refusal(tmp_path,attribute,value):
    path=write_results(tmp_path/"bad.h5",native())
    with h5py.File(path,"r+") as h:h["components/0/responses"].attrs[attribute]=value
    with pytest.raises(ValidationError):open_results(path)


def test_codec_association_tamper_refused_and_real_response_checksum(tmp_path):
    path=write_results(tmp_path/"association.h5",native())
    with h5py.File(path,"r+") as h:h["components/0/associations"][0,1]=99
    with open_results(path) as reader:
        with pytest.raises(ValidationError,match="binding|associations"):
            reader.read_field("a","stress",reader.components()[0]["component"])
    path=write_results(tmp_path/"response.h5",native(),compression=None)
    corrupt_payload(path,"components/0/responses",(0,0,0))
    with open_results(path) as reader:
        assert len(reader.cases())==2
        assert_checksum_error(lambda:reader.read_field("a","stress",reader.components()[0]["component"]))


def test_rehashed_duplicate_group_metadata_refused(tmp_path):
    from anypost.codec import association_digest
    path=write_results(tmp_path/"duplicate-group.h5",native())
    with h5py.File(path,"r+") as h:
        group=h["components/0"];values=group["associations"][:];values[14:21,0]=values[0,0]
        group["associations"][:]=values
        group["responses"].attrs["association_sha256"]=association_digest(values)
    with open_results(path,working_bytes=32768) as reader:
        with pytest.raises(ValidationError,match="unique"):
            reader.read_field("a","stress",reader.components()[0]["component"])


def test_actual_s3_seven_sample_seven_case_producer_roundtrip(tmp_path):
    from anysolver import PatternCase,RecoveryConfig,prepare_static_response_evaluator,export_recovery_capsule,export_recovery_capsule_to_sink
    from test_prepared_static_evaluator import _shell_model,_shell_case
    import anysolver_results as runtime
    cases=[PatternCase("fallback",{"p":1.},canonical_load_case=_shell_case(3,pressure=5).load_case)]
    cases += [PatternCase(f"physical-{i}",{"p":i/8}) for i in range(1,7)]
    recovery=RecoveryConfig(include_stresses=True,include_reactions=True)
    path=tmp_path/"actual-s3.h5"
    with prepare_static_response_evaluator(_shell_model(),{"p":_shell_case(1,pressure=2).load_case},recovery_config=recovery,stress_execution="certified_library") as evaluator:
        capsule=export_recovery_capsule(evaluator,cases,model_identity="actual-s3",mesh_identity="two-triangles-seven-samples")
        with ArchiveSink(path) as sink:
            export_recovery_capsule_to_sink(evaluator,cases,sink=sink,model_identity="actual-s3",mesh_identity="two-triangles-seven-samples",batch_size=2)
    descriptors={d["key"]:d for d in capsule["descriptors"]}
    def expected(ci,key):
        entry=capsule["stored"].get(key)
        if entry is not None and ci in entry["case_indices"]:
            row=np.flatnonzero(entry["case_indices"]==ci)[0]
            return entry["values"][row]
        if key in capsule["responses"]:
            active=capsule["cases"][ci]["active_indices"]
            weights=capsule["coefficients"][ci][np.asarray(active,dtype=int)-1]
            return runtime.reconstruct_response(capsule["responses"][key],weights,active_indices=active)
        spec=capsule["derived"][key]
        return runtime.derive_field(spec,{name:expected(ci,dep) for name,dep in spec["inputs"].items()})
    encoded=0
    with h5py.File(path) as h:
        meta=json.loads(h["metadata_json"][()].decode())
        for key,name in meta["component_groups"].items():
            group=h[f"components/{name}"]
            if "responses" in group:
                ds=group["responses"]
                if ds.ndim==3:
                    encoded+=1;assert ds.shape[1]>1
                    if key in ("membrane_xx","membrane_yy","membrane_xy","bending_xx","bending_yy","bending_xy","shear_xz","shear_yz"):assert ds.shape[1]==7
                    np.testing.assert_array_equal(ds[:].reshape(-1,ds.shape[-1]),capsule["responses"][key])
    assert encoded>0
    with open_results(path,cache_bytes=4096,working_bytes=65536) as reader:
        assert len(reader.cases())==7 and reader.cases()==tuple(capsule["cases"])
        for key,d in descriptors.items():
            if d["availability"]=="unavailable":continue
            np.testing.assert_array_equal(reader.associations(d["quantity_id"],d["component"]),capsule["associations"][key])
            for ci,case_record in enumerate(capsule["cases"]):
                np.testing.assert_allclose(reader.read_field(case_record["case_id"],d["quantity_id"],d["component"]),expected(ci,key),rtol=1e-12,atol=1e-12)
        assert reader.diagnostics["working_peak_bytes"]<=65536 and reader.diagnostics["cache_peak_bytes"]<=4096


def test_fortran_order_input_exact_decoder_and_unknown_three_dimensional_layout(tmp_path):
    c=native();c["responses"]={key:np.asfortranarray(value) for key,value in c["responses"].items()}
    path=write_results(tmp_path/"fortran.h5",c)
    with h5py.File(path,"r") as h:
        metadata=json.loads(h["metadata_json"][()].decode())
        for key,value in c["responses"].items():
            ds=h[f"components/{metadata['component_groups'][key]}/responses"]
            for station in range(7):np.testing.assert_array_equal(ds[:,station,:],value[station::7])
    with open_results(path,cache_bytes=256,working_bytes=8192) as reader:
        np.testing.assert_array_equal(reader.read_field("a","stress","vm",[34,0,34]),reference(c,"vm",selection=[34,0,34]))
    with h5py.File(path,"r+") as h:del h["components/0/responses"].attrs["codec"]
    with pytest.raises(ValidationError):open_results(path)
