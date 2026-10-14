"""Exact physical mask layout/selection tests; no changed engineering arithmetic."""
import json
import h5py
import numpy as np
import pytest
from anypost import write_results,open_results,ArchiveSink,ValidationError
from anypost.stored_codec import STORED_CODEC,NATIVE_STORED_CODEC,stored_shape,predictor_bits
from test_stored_correction import capsule,bits_equal,stream


def flatten_mask(ds):return ds[:].reshape(ds.shape[0],-1)


def test_native_masks_exact_uniform_grouping_and_old_flat_compatibility(tmp_path):
    c=capsule(n=210);native=write_results(tmp_path/"native.h5",c)
    old=tmp_path/"old.h5"
    with h5py.File(native) as source,h5py.File(old,"w") as target:
        for name,value in source.attrs.items():target.attrs[name]=value
        for name in source:source.copy(name,target)
    with h5py.File(old,"r+") as h:
        group=h["components/0"];bits=flatten_mask(group["values"]);del group["values"]
        ds=group.create_dataset("values",data=bits,chunks=(1,70),shuffle=True,fletcher32=True,compression="lzf");ds.attrs["codec"]=STORED_CODEC
    with h5py.File(native) as h:
        ds=h["components/0/values"];assert ds.shape==(6,30,7) and ds.chunks[0]==ds.chunks[2]==1
        assert ds.attrs["codec"]==NATIVE_STORED_CODEC and ds.attrs["association_sha256"]==h["components/0/responses"].attrs["association_sha256"]
    for selection in (slice(1,None,7),np.arange(2,210,14),np.array([209,0,209,7,8,0]),slice(None,None,-1),np.array([],dtype=int)):
        with open_results(native,cache_bytes=0,working_bytes=8192) as new,open_results(old,cache_bytes=0,working_bytes=8192) as prior:
            bits_equal(new.read_field("c1","stress","x",selection),prior.read_field("c1","stress","x",selection))
            assert new.diagnostics["working_peak_bytes"]<=8192


@pytest.mark.parametrize("mutation",["unknown","samples","entities","logical","digest","shape","flat_response"])
def test_native_source_layout_must_match_proven_response(tmp_path,mutation):
    path=write_results(tmp_path/"bad.h5",capsule())
    with h5py.File(path,"r+") as h:
        ds=h["components/0/values"]
        if mutation=="unknown":ds.attrs["codec"]="xor_ordered_ieee754_uniform_samples/9"
        elif mutation=="samples":ds.attrs["samples_per_entity"]=6
        elif mutation=="entities":ds.attrs["entity_count"]=20
        elif mutation=="logical":ds.attrs["logical_rows"]=69
        elif mutation=="digest":ds.attrs["association_sha256"]="0"*64
        elif mutation=="shape":
            attrs=dict(ds.attrs);del h["components/0/values"];ds=h["components/0"].create_dataset("values",shape=(6,7,10),dtype="uint64");ds.attrs.update(attrs)
        else:
            group=h["components/0"];values=group["responses"][:].reshape(70,4);del group["responses"];group.create_dataset("responses",data=values)
    with pytest.raises(ValidationError):open_results(path)


def test_station_hyperslabs_do_not_read_other_sample_chunks(tmp_path,monkeypatch):
    path=write_results(tmp_path/"station.h5",capsule(2100));reads=[]
    original=h5py.Dataset.__getitem__
    def traced(ds,selection):
        if ds.name.endswith("/values"):reads.append(selection)
        return original(ds,selection)
    monkeypatch.setattr(h5py.Dataset,"__getitem__",traced)
    with open_results(path,cache_bytes=0,working_bytes=65536) as reader:
        value=reader.read_field("c0","stress","x",np.arange(1,2100,7))
    bits_equal(value,capsule(2100)["stored"]["sx"]["values"][0,1::7])
    assert reads and all(len(selection)==3 and selection[0]==0 and isinstance(selection[1],slice) and selection[2]==1 for selection in reads)


def test_predictor_f_packing_retained_under_cache_and_exact_for_extremes(tmp_path):
    c=capsule(extreme=True);path=write_results(tmp_path/"f.h5",c)
    with open_results(path,cache_bytes=4096,working_bytes=65536) as reader:
        bits_equal(reader.read_field("c0","stress","x",slice(1,None,7)),c["stored"]["sx"]["values"][0,1::7])
        entries=[item[0] for key,item in reader._cache._values.items() if key[0]=="stored_predictor_basis"]
        assert entries and all(value.flags.f_contiguous and not value.flags.writeable for value in entries)
        assert reader.diagnostics["cache_peak_bytes"]<=4096
    indices=np.array([69,0,69,2,1,5]);weights=c["coefficients"][0]
    assert np.array_equal(predictor_bits(np.ascontiguousarray(c["responses"]["sx"][indices]),weights,[3,2,1]),predictor_bits(np.asfortranarray(c["responses"]["sx"][indices]),weights,[3,2,1]))


def test_plain_many_case_rows_are_not_multicase_chunks(tmp_path):
    c=capsule(7,native=False);count=1000
    c["cases"]=tuple(dict(case_id=f"p{i}",status="converged",certified=True,route="stored") for i in range(count))
    c["coefficients"]=np.zeros((count,3));values=np.arange(count*7,dtype=float).reshape(count,7)
    c["stored"]["sx"]={"case_indices":np.arange(count),"values":values}
    path=write_results(tmp_path/"plain.h5",c,stored_encoding="plain")
    with h5py.File(path) as h:
        ds=h["components/0/values"];assert ds.chunks==(1,7) and ds.id.get_num_chunks()==count
        assert np.prod(ds.chunks)*8==7*8
    with open_results(path) as reader:bits_equal(reader.read_field("p999","stress","x",[6,0,6]),values[999,[6,0,6]])


def test_pooled_native2_metadata_and_new_masks_coexist(tmp_path):
    c=capsule();c["cases"]=tuple(dict(case,reaction_node_ids=list(range(300))) for case in c["cases"])
    path=write_results(tmp_path/"pooled.h5",c)
    with h5py.File(path) as h:assert h.attrs["version"]==2 and h["components/0/values"].attrs["codec"]==NATIVE_STORED_CODEC
    with open_results(path) as reader:
        assert reader.case_metadata("c0")["reaction_node_ids"]==list(range(300))
        bits_equal(reader.read_field("c1","stress","x",[69,0,69]),c["stored"]["sx"]["values"][1,[69,0,69]])


def test_one_native_station_does_not_touch_other_corrupt_chunks(tmp_path):
    c=capsule();path=write_results(tmp_path/"other-station.h5",c,compression=None)
    with h5py.File(path) as h:offset=h["components/0/values"].id.get_chunk_info_by_coord((0,0,6)).byte_offset
    with path.open("r+b") as f:
        f.seek(offset);value=f.read(1);f.seek(offset);f.write(bytes([value[0]^64]))
    with open_results(path) as reader:
        bits_equal(reader.read_field("c0","stress","x",slice(1,None,7)),c["stored"]["sx"]["values"][0,1::7])
        with pytest.raises(ValidationError) as error:reader.read_field("c0","stress","x")
        assert isinstance(error.value.__cause__,OSError)


def test_derived_target_is_capped_by_unchanged_complete_cost_and_working_budget():
    from anypost.archive import ResultsReader,COMPUTE_BYTES
    from types import SimpleNamespace
    assert COMPUTE_BYTES==64*1024**2
    for working in (4096,256*1024**2):
        reader=SimpleNamespace(working_bytes=working,_meta={"derived":{"p":{}}})
        cost=6576;reserve=128
        tile=ResultsReader._tile_size(reader,"p",reserve,cost)
        if working>cost+reserve:assert reserve+tile*cost<=working and tile*cost<=COMPUTE_BYTES
