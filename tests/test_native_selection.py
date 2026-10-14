import h5py
import numpy as np
import pytest
from types import SimpleNamespace
from anypost import open_results,write_results
import anysolver_results as runtime
from test_derived_caching import shell_capsule,reference


@pytest.mark.parametrize("sample_count",[4,7])
def test_only_requested_native_samples_reconstructed_and_strided_hyperslabs(tmp_path,sample_count):
    entities=700;n=entities*sample_count;c=shell_capsule(n)
    associations=np.column_stack((np.repeat(np.arange(entities)+100,sample_count),np.tile(np.arange(sample_count),entities)))
    for key in c["associations"]:c["associations"][key]=associations.copy()
    path=write_results(tmp_path/"samples.h5",c);selection=np.arange(0,n,sample_count)
    expected=reference(c,"principal",selection=selection)
    calls={"primitive_rows":0,"derived_rows":0,"selectors":[]}
    with open_results(path,working_bytes=65536,cache_bytes=32768) as reader:
        original_dataset=reader._component_dataset
        class Proxy:
            def __init__(self,dataset):self.dataset=dataset
            def __getattr__(self,name):return getattr(self.dataset,name)
            def __getitem__(self,selector):
                calls["selectors"].append(selector);return self.dataset[selector]
        def dataset(key,name):
            value=original_dataset(key,name)
            return Proxy(value) if name=="responses" else value
        reader._component_dataset=dataset
        def reconstruct(prepared,coefficients,*,combination=None):
            calls["primitive_rows"]+=len(prepared.baseline)
            return runtime.reconstruct_prepared_response(prepared,coefficients,combination=combination)
        def derive(spec,inputs):
            calls["derived_rows"]+=len(next(iter(inputs.values())))
            return runtime.derive_field(spec,inputs)
        reader._runtime=SimpleNamespace(RUNTIME_ID=runtime.RUNTIME_ID,prepare_response=runtime.prepare_response,reconstruct_prepared_response=reconstruct,derive_field=derive,reconstruct_response=runtime.reconstruct_response)
        np.testing.assert_array_equal(reader.read_field("a","stress","principal",selection),expected)
        assert calls["derived_rows"]==entities and calls["primitive_rows"]==8*entities
        assert calls["selectors"] and all(isinstance(sl,tuple) and sl[1]==0 and isinstance(sl[0],slice) for sl in calls["selectors"])
        np.testing.assert_array_equal(reader.associations("stress","principal",selection),associations[selection])
        assert reader.diagnostics["working_peak_bytes"]<=65536
        assert reader.diagnostics["cache_peak_bytes"]<=32768
        # Eviction must preserve values/selection order, not expand source samples.
        reader.clear_cache()
        np.testing.assert_array_equal(reader.read_field("b","stress","principal",selection),reference(c,"principal",case=1,selection=selection))
        assert calls["derived_rows"]==2*entities and calls["primitive_rows"]==16*entities


@pytest.mark.parametrize("selection",[np.array([14,0,14,7,0,20]),np.arange(0,1400,7)[::-1],slice(0,1400,7)])
def test_native_reordered_duplicate_reverse_and_slice_runtime_parity(tmp_path,selection):
    c=shell_capsule(1400);path=write_results(tmp_path/"ordering.h5",c)
    with open_results(path,working_bytes=65536,cache_bytes=16384) as reader:
        for key in ("vm","principal"):
            np.testing.assert_array_equal(reader.read_field("a","stress",key,selection),reference(c,key,selection=selection))
        assert reader.diagnostics["working_peak_bytes"]<=65536
        assert reader.diagnostics["cache_peak_bytes"]<=16384


def test_stored_row_regular_stride_preserves_values_and_associations(tmp_path):
    c=shell_capsule(1400)
    c["descriptors"]=tuple(dict(d,availability="stored") for d in c["descriptors"][:1])
    key=c["descriptors"][0]["key"]
    c["responses"]={};c["derived"]={};c["associations"]={key:c["associations"][key]}
    values=np.arange(2800,dtype=float).reshape(2,1400)
    c["stored"]={key:{"case_indices":np.array([0,1]),"values":values}}
    path=write_results(tmp_path/"stored.h5",c);selected=np.arange(0,1400,7)
    with open_results(path) as reader:
        np.testing.assert_array_equal(reader.read_field("a","stress",key,selected),values[0,selected])
        np.testing.assert_array_equal(reader.associations("stress",key,selected),c["associations"][key][selected])
