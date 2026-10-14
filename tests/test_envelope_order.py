import copy
import json
import h5py
import numpy as np
import pytest
from anypost import write_results,open_results,ValidationError
from test_stream import header,case,descriptor


def envelope_capsule(policy):
    c=header(4)
    c.update(responses={},stored={"sx":{"case_indices":np.arange(4),"values":np.array([[1.,1.,1.,1.],[np.nextafter(1.,np.inf),1.,1.,np.nextafter(1.,-np.inf)],[1.,1.,1.,1.],[1.e100,1.e100,1.e100,1.e100]],dtype=np.float64)}},descriptors=(dict(descriptor(availability="stored"),tie_policy=policy),),cases=(dict(case("early"),input_index=9),dict(case("late"),input_index=31),dict(case("middle"),input_index=17),dict(case("failed"),input_index=0,status="failed",certified=False)),coefficients=np.zeros((4,2)))
    return c


@pytest.mark.parametrize("policy,tie_case",[("first","late"),("last","early"),("exact_earliest_input_index","early")])
def test_near_tie_exact_tie_and_failed_case_governing_ids(tmp_path,policy,tie_case):
    path=write_results(tmp_path/"envelope.h5",envelope_capsule(policy))
    with open_results(path,cache_bytes=256,working_bytes=4096) as reader:
        env=reader.read_envelope("stress","sx",case_ids=["late","middle","failed","early"])
        np.testing.assert_array_equal(env.maximum,[np.nextafter(1.,np.inf),1.,1.,1.])
        np.testing.assert_array_equal(env.minimum,[1.,1.,1.,np.nextafter(1.,-np.inf)])
        assert env.max_case_ids.tolist()==["late",tie_case,tie_case,"middle" if policy=="first" else "early"]
        assert env.min_case_ids.tolist()==["early" if policy!="first" else "middle",tie_case,tie_case,"late"]
        assert reader.cases()[-1]["case_id"]=="failed"
        if policy=="exact_earliest_input_index":
            reversed_subset=reader.read_envelope("stress","sx",case_ids=["middle","early"])
            assert reversed_subset.min_case_ids.tolist()==["early"]*4
            assert reversed_subset.max_case_ids.tolist()==["early"]*4


@pytest.mark.parametrize("indices",[(None,31,17,0),(9,9,17,0),(-1,31,17,0),(True,31,17,0),(9,3.5,17,0)])
def test_authoritative_index_policy_invalid_metadata_refused(tmp_path,indices):
    c=envelope_capsule("exact_earliest_input_index")
    c["cases"]=tuple(dict(record,input_index=index) for record,index in zip(c["cases"],indices))
    with pytest.raises(ValidationError,match="input_index"):write_results(tmp_path/"bad.h5",c)
    valid=envelope_capsule("exact_earliest_input_index");path=write_results(tmp_path/"corrupted.h5",valid)
    with h5py.File(path,"r+") as h:
        metadata=json.loads(h["metadata_json"][()].decode())
        metadata["cases"]=list(c["cases"])
        h["metadata_json"][()]=json.dumps(metadata)
    with pytest.raises(ValidationError,match="input_index"):open_results(path)
