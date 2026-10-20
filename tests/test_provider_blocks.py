"""Dense switching follows native row arithmetic and provider boundary rules."""
import numpy as np
import pytest

from anypost import (open_memory_results, reduce_samples, RequestCancelled,
                     ValidationError, WorkingSetError)


@pytest.mark.parametrize('method', ['mean', 'min', 'max', 'max_abs'])
@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'fortran', 'reversed'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_rectangular_reduction_is_bitwise_native_row_arithmetic(method, layout, dtype):
    values = np.random.default_rng(7).normal(size=(4103, 17)).astype(dtype)
    values[0] = 0.; values[1] = -0.
    values[2, :2] = [-100., 100.]  # Signed first max_abs tie.
    if layout == 'strided': values = values[:, ::2]
    elif layout == 'fortran': values = np.asfortranarray(values)
    elif layout == 'reversed': values = values[::-1, ::-1]
    if method == 'max_abs':
        expected = np.array([row[np.argmax(np.abs(row))] for row in values], dtype=dtype)
    else:
        expected = np.array([getattr(np, method)(row) for row in values], dtype=dtype)
    actual = reduce_samples(values, method)
    assert actual.tobytes() == expected.tobytes()
    with pytest.raises(ValueError): actual.setflags(write=True)


class DenseProvider:
    def __init__(self, n=9001):
        self.values = np.arange(n * 3, dtype=np.float64).reshape(n, 3)
        self.calls = 0
    def metadata(self):
        return dict(model_identity='dense', cases=[dict(id='a', status='completed')],
                    descriptors=[dict(quantity_id='stress', components=('xx',),
                        location='element', spatial_count=len(self.values),
                        value_shape=(3,), provider_work_bytes=0)])
    def read_field(self, *args, selection=None, cancel_event=None):
        self.calls += 1
        return self.values[selection]


def test_provider_dense_refusal_cache_mutation_and_cancel_contracts():
    provider = DenseProvider()
    with open_memory_results(provider) as reader:
        provider.values[-1, -1] = np.inf
        with pytest.raises(ValidationError, match='infinite'):
            reader.read_field('a', 'stress', 'xx', reduction='mean')
        provider.values[-1, -1] = np.nan
        with pytest.raises(ValidationError, match='NaN'):
            reader.read_field('a', 'stress', 'xx', nan_policy='raise')
        provider.values[-1, -1] = 1.
        result = reader.read_field('a', 'stress', 'xx', reduction='mean')
        before = result.tobytes(); calls = provider.calls
        provider.values[:] = -99.
        assert reader.read_field('a', 'stress', 'xx', reduction='mean').tobytes() == before
        assert provider.calls == calls
        reader.clear_cache()
        assert np.all(reader.read_field('a', 'stress', 'xx', reduction='mean') == -99.)


class CancelAfter:
    def __init__(self, calls): self.calls = calls
    def is_set(self):
        self.calls -= 1
        return self.calls <= 0


def test_block_cancellation_does_not_publish_partial_cache():
    provider = DenseProvider()
    with open_memory_results(provider) as reader:
        with pytest.raises(RequestCancelled):
            reader.read_field('a', 'stress', 'xx', cancel_event=CancelAfter(6))
        before = provider.calls
        reader.read_field('a', 'stress', 'xx')
        assert provider.calls == before + 1
    with pytest.raises(RequestCancelled):
        reduce_samples(provider.values, 'max_abs', cancel_event=CancelAfter(2))


def test_dense_reduction_keeps_working_cap_refusal():
    with pytest.raises(WorkingSetError):
        reduce_samples(np.ones((9001, 3)), 'mean', working_bytes=4096)


@pytest.mark.parametrize('dtype', [np.float16, np.float32])
@pytest.mark.parametrize('shape', [(1, 1), (2, 2), (4103, 17)])
@pytest.mark.parametrize('order', ['C', 'F'])
def test_small_dtype_max_abs_preserves_first_ties_at_existing_working_bound(dtype, shape, order):
    values = np.ones(shape, dtype=dtype, order=order)
    values[:, 0] = -1.
    n, width = shape
    budget = max(values.nbytes*2+n*24+4096,
                 values.nbytes*2+n*8+n*values.dtype.itemsize*2+
                 width*(values.dtype.itemsize+1)+4096)
    result = reduce_samples(values, 'max_abs', working_bytes=budget)
    assert result.dtype == dtype
    assert np.all(result == -1.)


@pytest.mark.parametrize('method', ['mean', 'min', 'max', 'max_abs'])
def test_multiaxis_reshape_copy_keeps_native_arithmetic_and_existing_bound(method):
    source = np.random.default_rng(23).normal(size=(37, 8, 13))
    values = source[:, ::2, ::2]
    rows = values.reshape(len(values), -1)
    assert not np.may_share_memory(values, rows)
    width = rows.shape[1]; n = len(values)
    budget = max(values.nbytes*2+n*24+4096,
                 values.nbytes*2+n*8+n*values.dtype.itemsize*2+
                 width*(values.dtype.itemsize+1)+4096)
    expected = np.array([row[np.argmax(np.abs(row))] if method=='max_abs'
                         else getattr(np,method)(row) for row in rows])
    actual = reduce_samples(values, method, working_bytes=budget)
    assert actual.tobytes() == expected.tobytes()
