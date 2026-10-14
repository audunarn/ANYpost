import copy,gc,json,weakref
from queue import Queue
import h5py
import numpy as np
import pytest
from anypost import write_results,open_results,ArchiveSink,ValidationError,FailedCaseError,WorkingSetError
from anypost.stored_codec import predictor_bits,STORED_CODEC,NATIVE_STORED_CODEC,CANONICAL_NAN,ENCODE_WORK_BYTES


def capsule(n=70,native=True,extreme=False):
    rng=np.random.default_rng(710)
    basis=np.asfortranarray(rng.normal(size=(n,4)))
    basis[0,0]=-0.
    coefficients=np.array([[.3,-.7,1.1],[.5,-.25,3.],[0.,2.,0.],[0.,0.,0.],[.1,.2,.3],[4.,-2.,1.]])
    if extreme:
        basis[:,1]=1e308;basis[:,2]=-1e308;basis[:,3]=np.nextafter(0.,1.)
        coefficients[0]=[1e308,1e308,.5]
    active=(None,[3,1,2],[2],[],[1,2,3],[2,1])
    cases=tuple(dict(case_id=f"c{i}",status="failed" if i==5 else "converged",certified=i not in (4,5),route="source-stored",active_indices=indices) for i,indices in enumerate(active))
    actual=rng.normal(size=(6,n));actual[:,0]=-0.;actual[:,1]=0.;actual[:,2]=np.finfo(float).max;actual[:,3]=np.nextafter(0.,1.)
    associations=np.column_stack((np.repeat(np.arange(n//7)*13+100,7),np.tile(np.arange(7),n//7))) if native else np.arange(n)
    return dict(schema="anysolver.recovery_capsule",version=1,runtime="anysolver-results/1",model_identity="exact-correction",mesh_identity="explicit-source-rows",solver_identity={},geometry={},patterns=("p1","p2","p3"),cases=cases,coefficients=coefficients,descriptors=(dict(key="sx",quantity_id="stress",component="x",availability="reconstructable",location="element",unit="Pa",basis="source",sampling="source-native"),),associations={"sx":associations},responses={"sx":basis},stored={"sx":{"case_indices":np.arange(6),"values":actual}},derived={})


def bits_equal(a,b):np.testing.assert_array_equal(np.asarray(a).view(np.uint64),np.asarray(b).view(np.uint64))


@pytest.mark.parametrize("compression",[None,"lzf"])
@pytest.mark.parametrize("native",[True,False])
@pytest.mark.parametrize("encoding",["auto","plain"])
def test_bitwise_roundtrip_all_active_orders_and_source_signed_zero(tmp_path,compression,native,encoding,monkeypatch):
    c=capsule(native=native);path=write_results(tmp_path/"roundtrip.h5",c,compression=compression,stored_encoding=encoding)
    with h5py.File(path) as h:
        ds=h["components/0/values"];assert ds.dtype==np.dtype("uint64" if encoding=="auto" else "float64")
        assert ds.attrs.get("codec")== (NATIVE_STORED_CODEC if native and encoding=="auto" else STORED_CODEC if encoding=="auto" else None)
        if encoding=="auto":assert ds.chunks[0]==1 and np.prod(ds.chunks)*8<=256*1024 and ds.fletcher32 and ds.shuffle
    import anypost.archive as archive
    monkeypatch.setattr(archive.importlib,"import_module",lambda name: (_ for _ in ()).throw(AssertionError("stored codec imported engineering runtime")))
    with open_results(path,cache_bytes=512,working_bytes=8192) as reader:
        assert reader.components()[0]["availability"]==c["descriptors"][0]["availability"]
        assert len(reader._stored_rows["sx"])==6
        for ci in range(5):
            for selection in (None,np.array([69,0,69,1,2,3]),slice(None,None,-1),np.arange(0,70,7),np.array([],dtype=int)):
                chosen=np.arange(70) if selection is None else np.arange(70)[selection]
                actual=reader.read_field(f"c{ci}","stress","x",selection)
                bits_equal(actual,c["stored"]["sx"]["values"][ci,chosen]);assert not actual.flags.writeable
        with pytest.raises(FailedCaseError):reader.read_field("c5","stress","x")
        assert reader.diagnostics["cache_peak_bytes"]<=512 and reader.diagnostics["working_peak_bytes"]<=8192
        assert reader.diagnostics["prepared_reconstructions"]==0 and reader.diagnostics["reconstructions"]==0
        reader.clear_cache();bits_equal(reader.read_field("c1","stress","x",[69,0,69]),c["stored"]["sx"]["values"][1,[69,0,69]])


def test_extreme_finite_predictor_nan_portability_and_local_errstate(tmp_path):
    c=capsule(extreme=True)
    original=np.geterr()
    np.seterr(over="raise",under="raise",invalid="raise")
    expected=np.geterr()
    try:
        predicted=predictor_bits(c["responses"]["sx"],c["coefficients"][0],None)
        assert np.all(predicted==CANONICAL_NAN)
        baseline=predictor_bits(c["responses"]["sx"],c["coefficients"][3],[])
        bits_equal(baseline.view(np.float64),c["responses"]["sx"][:,0])
        path=write_results(tmp_path/"extreme.h5",c)
        with open_results(path,cache_bytes=0) as reader:
            bits_equal(reader.read_field("c0","stress","x",[69,0,69,3]),c["stored"]["sx"]["values"][0,[69,0,69,3]])
        assert np.geterr()==expected
    finally:np.seterr(**original)


def stream(path,c,encoding="auto",abort=False):
    h={key:value for key,value in c.items() if key not in ("cases","coefficients","descriptors","stored")}
    sink=ArchiveSink(path,stored_encoding=encoding)
    with sink:
        sink.capsule_begin(h)
        for i,case in enumerate(c["cases"]):sink.capsule_case(case,c["coefficients"][i])
        for first in (0,2,4):
            rows=c["stored"]["sx"]["values"][first:first+2].copy();rows.setflags(write=False);ref=weakref.ref(rows)
            sink.capsule_stored("sx",np.arange(first,first+2),rows)
            del rows;gc.collect();assert ref() is None
        if abort:raise RuntimeError("cancelled source export")
        sink.capsule_end(dict(descriptors=c["descriptors"],associations=c["associations"],responses=c["responses"],derived={},case_count=6,stored_superseded_cases={}))
    return sink


@pytest.mark.parametrize("encoding",["auto","plain"])
def test_stream_exact_batch_append_no_payload_retention_atomic_cancel(tmp_path,encoding):
    c=capsule(2100);path=tmp_path/"stream.h5"
    for value in (c["responses"]["sx"],c["coefficients"],c["stored"]["sx"]["values"]):value.setflags(write=False)
    sink=stream(path,c,encoding)
    assert sink.diagnostics["retained_explicit_value_bytes"]==0
    assert sink.diagnostics["peak_encode_work_bytes"]<=ENCODE_WORK_BYTES
    assert sink.diagnostics["peak_write_block_bytes"]<=256*1024
    selection=np.array([2099,0,2099,1,2,3])
    with open_results(path,cache_bytes=256,working_bytes=4096) as reader:
        bits_equal(reader.read_field("c0","stress","x",selection),c["stored"]["sx"]["values"][0,selection])
    original=path.read_bytes()
    with pytest.raises(RuntimeError):stream(path,c,encoding,abort=True)
    assert path.read_bytes()==original and not list(tmp_path.glob(".*.tmp"))


def test_stream_late_response_keeps_existing_plain_data(tmp_path):
    c=capsule();h={key:value for key,value in c.items() if key not in ("cases","coefficients","descriptors","stored")};h["responses"]={}
    path=tmp_path/"late.h5"
    with ArchiveSink(path) as sink:
        sink.capsule_begin(h)
        for i,case in enumerate(c["cases"]):sink.capsule_case(case,c["coefficients"][i])
        sink.capsule_stored("sx",np.arange(6),c["stored"]["sx"]["values"])
        sink.capsule_end(dict(descriptors=c["descriptors"],associations=c["associations"],responses=c["responses"],derived={},case_count=6,stored_superseded_cases={}))
    with h5py.File(path) as h:assert h["components/0/values"].dtype==np.dtype("float64")
    with open_results(path) as reader:bits_equal(reader.read_field("c0","stress","x"),c["stored"]["sx"]["values"][0])


@pytest.mark.parametrize("mutation",["unknown","missing_tag","missing_basis","missing_coefficients","bad_active","bad_basis_shape"])
def test_missing_unknown_incompatible_codec_inputs_refused(tmp_path,mutation):
    path=write_results(tmp_path/"bad.h5",capsule())
    with h5py.File(path,"r+") as h:
        if mutation=="unknown":h["components/0/values"].attrs["codec"]="untrusted/4"
        elif mutation=="missing_tag":del h["components/0/values"].attrs["codec"]
        elif mutation=="missing_basis":del h["components/0/responses"]
        elif mutation=="missing_coefficients":del h["coefficients"]
        elif mutation=="bad_basis_shape":h["components/0/responses"].attrs["samples_per_entity"]=4
        else:
            meta=json.loads(h["metadata_json"][()].decode());meta["cases"][0]["active_indices"]=[0];h["metadata_json"][()]=json.dumps(meta)
    with pytest.raises(ValidationError):open_results(path)


def test_real_mask_checksum_and_nonfinite_decoded_value_refusal(tmp_path):
    c=capsule();path=write_results(tmp_path/"checksum.h5",c,compression=None)
    with h5py.File(path) as h:ds=h["components/0/values"];chunk=ds.id.get_chunk_info_by_coord((0,)*ds.ndim);offset=chunk.byte_offset
    with path.open("r+b") as f:f.seek(offset);byte=f.read(1);f.seek(offset);f.write(bytes([byte[0]^64]))
    with open_results(path) as reader:
        assert len(reader.cases())==6
        with pytest.raises(ValidationError) as error:reader.read_field("c0","stress","x")
        assert isinstance(error.value.__cause__,OSError)
    path=write_results(tmp_path/"invalid-decoded.h5",c)
    predicted=predictor_bits(c["responses"]["sx"][:1],c["coefficients"][0],None)
    with h5py.File(path,"r+") as h:
        ds=h["components/0/values"];ds[(0,)*ds.ndim]=np.array([np.inf]).view(np.uint64)[0]^predicted[0]
    with open_results(path) as reader:
        with pytest.raises(ValidationError,match="nonfinite"):reader.read_field("c0","stress","x",[0])


def test_work_cap_eviction_and_cancelled_completion_release(tmp_path):
    c=capsule(2100);path=write_results(tmp_path/"bounded.h5",c)
    selected=np.arange(0,2100,7)
    with open_results(path,cache_bytes=512,working_bytes=12288) as reader:
        bits_equal(reader.read_field("c1","stress","x",selected),c["stored"]["sx"]["values"][1,selected])
        reader.clear_cache()
        bits_equal(reader.read_field("c2","stress","x",selected[::-1]),c["stored"]["sx"]["values"][2,selected[::-1]])
        assert reader.diagnostics["cache_peak_bytes"]<=512 and reader.diagnostics["working_peak_bytes"]<=12288
    with open_results(path,cache_bytes=0,working_bytes=512) as reader:
        with pytest.raises(WorkingSetError):reader.read_field("c0","stress","x",selected)
        assert reader.diagnostics["reads"]==0
    from anypost.async_loader import LatestRequestLoader
    from threading import Event
    queue=Queue();published=[];loaded=Event();refs=[]
    with open_results(path,cache_bytes=0) as reader:
        loader=LatestRequestLoader(queue.put)
        def load(cancel):
            values=reader.read_field("c0","stress","x",[0,7,14]);refs.append(weakref.ref(values));loaded.set();return values
        loader.request(load,published.append)
        assert loaded.wait(2);callback=queue.get(timeout=2);loader.close();callback();gc.collect()
        assert not published and refs[0]() is None


def test_writer_failure_preserves_previous_archive_and_options(tmp_path,monkeypatch):
    import anypost.archive as archive
    path=tmp_path/"old.h5";path.write_bytes(b"previous result")
    def fail(*args,**kwargs):raise RuntimeError("interrupted mask encoding")
    monkeypatch.setattr(archive,"write_corrections",fail)
    with pytest.raises(RuntimeError):write_results(path,capsule())
    assert path.read_bytes()==b"previous result" and not list(tmp_path.glob(".*.tmp"))
    for factory in (write_results,ArchiveSink):
        with pytest.raises(ValueError):factory(path,stored_encoding="bad",**({"capsule":capsule()} if factory is write_results else {}))


def test_helper_numpy_integer_active_order_and_tiled_canonical_predictor():
    c=capsule(extreme=True);basis=c["responses"]["sx"];weights=c["coefficients"][0]
    active=np.array([2,1,3],dtype=np.int64)
    expected=predictor_bits(basis,weights,active)
    for start in range(0,len(basis),9):
        np.testing.assert_array_equal(predictor_bits(basis[start:start+9],weights,active),expected[start:start+9])
    chosen=np.array([69,0,69,1,3])
    np.testing.assert_array_equal(predictor_bits(basis[chosen],weights,active),expected[chosen])
    for bad in ([True],[1.,2.],[-1],[1,1]):
        with pytest.raises(ValidationError):predictor_bits(basis,weights,bad)


def test_final_response_signed_zero_bit_change_is_rejected_atomically(tmp_path):
    c=capsule();header={key:value for key,value in c.items() if key not in ("cases","coefficients","descriptors","stored")}
    path=tmp_path/"bits.h5";path.write_bytes(b"previous")
    with pytest.raises(ValidationError,match="static values"):
        with ArchiveSink(path) as sink:
            sink.capsule_begin(header)
            for i,case in enumerate(c["cases"]):sink.capsule_case(case,c["coefficients"][i])
            sink.capsule_stored("sx",np.arange(6),c["stored"]["sx"]["values"])
            changed=c["responses"]["sx"].copy();changed[0,0]=0.
            sink.capsule_end(dict(descriptors=c["descriptors"],associations=c["associations"],responses={"sx":changed},derived={},case_count=6,stored_superseded_cases={}))
    assert path.read_bytes()==b"previous"
