import numpy as np
import pytest
from anypost import write_results,open_results
from test_derived_caching import shell_capsule


def fixture(tmp_path):
    c=shell_capsule(4200);c.update(version=1,runtime='anysolver-results/1');c.pop('combination',None)
    return write_results(tmp_path/'workers.h5',c)


@pytest.mark.parametrize('bad',[True,False,0,3,1.5,'2'])
def test_workers_validation_without_runtime_import(tmp_path,bad):
    with pytest.raises(ValueError,match='derivation_workers'):open_results(fixture(tmp_path),derivation_workers=bad)


@pytest.mark.parametrize('requested',[None,1,2])
@pytest.mark.parametrize('capability',[False,True])
def test_capability_optional_dispatch_diagnostics_and_conservative_budget(tmp_path,monkeypatch,requested,capability):
    path=fixture(tmp_path)
    import anysolver_results as runtime
    original=runtime.derive_field;calls=[]
    def derive(spec,inputs,**kwargs):
        calls.append(kwargs)
        if kwargs:kwargs['execution'].update(workers=kwargs['workers'] if len(next(iter(inputs.values())))>=4096 else 1,tensors=len(next(iter(inputs.values()))))
        return original(spec,inputs)
    monkeypatch.setattr(runtime,'DERIVATION_EXECUTION_API',1 if capability else 0,raising=False)
    monkeypatch.setattr(runtime,'derive_field',derive)
    with open_results(path,derivation_workers=requested,working_bytes=32<<20) as reader:
        assert reader._runtime is None and reader.diagnostics['derivation_workers_effective']==1
        a=reader.read_field('a','stress','principal')
        expected=2 if capability and requested in (None,2) else 1
        assert calls[0].keys()==({'workers','execution'} if capability else set())
        assert reader.diagnostics['derivation_workers_effective']==expected
        assert reader.diagnostics['derivation_parallel_calls']==(1 if expected==2 else 0)
        if expected==2:
            cost=reader._cost('principal',0)
            reader.derivation_workers=1
            assert cost-reader._cost('principal',0)==64
        assert reader.diagnostics['working_peak_bytes']<=reader.working_bytes
        reader.clear_cache();b=reader.read_field('a','stress','principal',[0,1,1])
        assert reader.diagnostics['derivation_workers_effective']==1
        np.testing.assert_array_equal(b,a[[0,1,1]])
