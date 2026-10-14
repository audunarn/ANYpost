"""Corrupt actual allocated Fletcher32 DATA payloads, never HDF5 headers."""
import json
import h5py
import numpy as np
import pytest
from anypost import write_results,open_results,ValidationError,ClosedReaderError
from test_archive import capsule
from test_legacy import legacy_fixture


def corrupt_payload(path,dataset_path,coordinate):
    with h5py.File(path,"r") as handle:
        dataset=handle[dataset_path]
        assert dataset.fletcher32
        info=dataset.id.get_chunk_info_by_coord(coordinate)
        assert info.byte_offset is not None and info.size>4
        # Last four bytes are the Fletcher32 trailer. Flip a payload byte in
        # the allocated data chunk, without touching any object/file header.
        offset=int(info.byte_offset)+(int(info.size)-4)//2
    with open(path,"r+b") as stream:
        stream.seek(offset);original=stream.read(1);assert len(original)==1
        stream.seek(offset);stream.write(bytes([original[0]^0x40]))
    return offset


def assert_checksum_error(operation):
    with pytest.raises(ValidationError) as caught:operation()
    assert isinstance(caught.value.__cause__,OSError)
    message=str(caught.value.__cause__).lower()
    assert any(token in message for token in ("fletcher", "checksum", "filter returned failure"))


def test_real_fletcher32_field_corruption_metadata_discovery_and_clean_close(tmp_path,capsule):
    path=write_results(tmp_path/"field.h5",capsule,compression=None)
    corrupt_payload(path,"components/0/values",(0,0))
    reader=open_results(path)
    assert reader.components()[0]["unit"]=="Pa"
    assert reader.cases()[2]["diagnostics"]==["failure retained"]
    assert reader.diagnostics["reads"]==0
    assert_checksum_error(lambda:reader.read_field("a","stress","x"))
    assert_checksum_error(lambda:reader.read_envelope("stress","x"))
    # The corruption is field-local: diagnostic inventory remains accessible.
    assert len(reader.cases())==3
    reader.close();reader.close()
    assert not reader._handle.id.valid
    with pytest.raises(ClosedReaderError):reader.read_field("a","stress","x")


@pytest.mark.parametrize("name",["coefficients","associations","geometry"])
def test_corrupt_coefficient_association_geometry_public_and_direct_typed_reads(tmp_path,capsule,name):
    path=write_results(tmp_path/f"{name}.h5",capsule,compression=None)
    target={"coefficients":("coefficients",(0,0)),"associations":("components/0/associations",(0,)),"geometry":("geometry/positions",(0,0))}[name]
    corrupt_payload(path,*target)
    with open_results(path) as reader:
        assert len(reader.components())==1
        if name=="coefficients":
            assert_checksum_error(lambda:reader._coefficient_row(0))
        elif name=="associations":
            assert_checksum_error(lambda:reader.associations("stress","x"))
            assert_checksum_error(lambda:reader._spatial_read(reader._component_dataset("sx","associations"),range(5)))
        else:
            assert_checksum_error(reader.geometry)
    assert not reader._handle.id.valid


def test_corrupt_response_and_coefficient_paths_refuse_before_success(tmp_path,capsule):
    c=capsule;c["descriptors"][0]["availability"]="reconstructable"
    c["stored"]={};c["responses"]={"sx":np.ones((5,2))}
    for target,coord in (("coefficients",(0,0)),("components/0/responses",(0,0))):
        path=write_results(tmp_path/(target.replace("/","-")+".h5"),c,compression=None)
        corrupt_payload(path,target,coord)
        with open_results(path) as reader:
            assert_checksum_error(lambda:reader.read_field("a","stress","x"))
            assert reader.diagnostics["prepared_reconstructions"]==0


def test_constructor_checksum_failure_closes_every_open_handle(tmp_path,capsule,monkeypatch):
    path=write_results(tmp_path/"metadata-row-index.h5",capsule,compression=None)
    corrupt_payload(path,"components/0/case_indices",(0,))
    original=h5py.File;opened=[]
    def tracking(*args,**kwargs):
        handle=original(*args,**kwargs);opened.append(handle);return handle
    monkeypatch.setattr(h5py,"File",tracking)
    assert_checksum_error(lambda:open_results(path))
    assert opened and all(not handle.id.valid for handle in opened)


def test_legacy_checksum_field_and_table_errors_are_typed(tmp_path):
    path=tmp_path/"legacy.h5";values=legacy_fixture(path)
    with h5py.File(path,"r+") as handle:
        del handle["fields/stress/values"]
        handle["fields/stress"].create_dataset("values",data=values,chunks=(1,3,2),shuffle=True,fletcher32=True)
        del handle["tables/stress_element_ids"]
        handle["tables"].create_dataset("stress_element_ids",data=[11,12,13],chunks=(3,),fletcher32=True)
    corrupt_payload(path,"fields/stress/values",(0,0,0))
    corrupt_payload(path,"tables/stress_element_ids",(0,))
    with open_results(path) as reader:
        assert len(reader.components())==2
        assert_checksum_error(lambda:reader.read_field("0","stress","x"))
        assert_checksum_error(lambda:reader.associations("stress","x"))
        assert_checksum_error(lambda:reader.table("stress_element_ids"))
    assert not reader._handle.id.valid
