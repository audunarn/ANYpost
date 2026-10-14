"""Exact stored-value byte codec. Prediction is never an engineering result."""
import numpy as np
from .errors import ValidationError, WorkingSetError
from .codec import response_shape, CODEC as RESPONSE_CODEC

STORED_CODEC = "xor_ordered_ieee754/1"
NATIVE_STORED_CODEC = "xor_ordered_ieee754_uniform_samples/1"
CANONICAL_NAN = np.uint64(0x7ff8000000000000)
ENCODE_WORK_BYTES = 8 * 1024**2
CHUNK_BYTES = 256 * 1024


def predictor_bits(basis, coefficients, active_indices):
    """Fixed separate float64 multiply/add ufuncs, original active order."""
    basis = np.asarray(basis, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if basis.ndim != 2 or coefficients.shape != (basis.shape[1]-1,):
        raise ValidationError("stored predictor basis/coefficient shape mismatch")
    active = range(1, basis.shape[1]) if active_indices is None else active_indices
    if any(not isinstance(column, (int, np.integer)) or isinstance(column, (bool, np.bool_)) or column < 1 or column >= basis.shape[1] for column in active) or len(set(active)) != len(active):
        raise ValidationError("invalid stored predictor active columns")
    # Empty activity preserves baseline bits, including negative zero.
    if not len(active):
        predicted = basis[:, 0].copy()
    else:
        predicted = np.zeros(len(basis), dtype=np.float64)
        scratch = np.empty(len(basis), dtype=np.float64)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            for column in active:
                np.multiply(basis[:, column], coefficients[column-1], out=scratch)
                np.add(predicted, scratch, out=predicted)
            np.add(predicted, basis[:, 0], out=predicted)
    bits = predicted.view(np.uint64)
    bits[np.isnan(predicted)] = CANONICAL_NAN
    return bits


def decode_stored(mask, basis, coefficients, active_indices):
    mask = np.asarray(mask)
    if mask.dtype != np.dtype("uint64") or mask.shape != (len(basis),):
        raise ValidationError("stored correction mask must be one uint64 per source row")
    bits = np.bitwise_xor(mask, predictor_bits(basis, coefficients, active_indices))
    actual = bits.view(np.float64)
    if not np.isfinite(actual).all():
        raise ValidationError("decoded stored engineering values contain nonfinite data")
    return actual


def stored_encoding_option(value):
    if value not in ("auto", "plain"):
        raise ValueError("stored_encoding must be 'auto' or 'plain'")
    return value


def stored_shape(ds, shape, has_basis, *, response=None):
    codec = ds.attrs.get("codec")
    if isinstance(codec, bytes):codec = codec.decode("utf-8")
    if codec is None:
        if ds.shape != shape or ds.dtype != np.dtype("float64"):
            raise ValidationError("unmarked stored data require matching float64 layout")
        return False
    if codec not in (STORED_CODEC,NATIVE_STORED_CODEC):
        raise ValidationError("unknown stored correction codec")
    if ds.dtype != np.dtype("uint64") or not has_basis:
        raise ValidationError("stored correction codec requires uint64 and compatible response basis")
    if codec == STORED_CODEC:
        if ds.shape != shape:raise ValidationError("invalid explicit value dataset")
    else:
        if response is None or response_shape(response,shape[1],response.shape[-1]) is None:
            raise ValidationError("native stored correction requires uniform response source")
        for name in ("samples_per_entity","entity_count","logical_rows"):
            value=ds.attrs.get(name)
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or int(value)!=int(response.attrs[name]):
                raise ValidationError("native stored correction source shape mismatch")
        digest=ds.attrs.get("association_sha256")
        if isinstance(digest,bytes):digest=digest.decode("utf-8")
        source_digest=response.attrs["association_sha256"]
        if isinstance(source_digest,bytes):source_digest=source_digest.decode("utf-8")
        if digest!=source_digest or ds.shape!=(shape[0],response.shape[0],response.shape[1]):
            raise ValidationError("native stored correction source association/layout mismatch")
    return True


def correction_dataset(group, rows, logical_rows, response, compression, *, resizable=False):
    step=encode_tile_rows(response.shape[-1])
    step=1 << (step.bit_length()-1)
    samples=response_shape(response,logical_rows,response.shape[-1])
    if samples is None:
        ds=group.create_dataset("values",shape=(rows,logical_rows),maxshape=(None,logical_rows) if resizable else None,dtype="uint64",chunks=(1,min(logical_rows,step)),compression=compression,shuffle=True,fletcher32=True)
        ds.attrs["codec"]=STORED_CODEC
    else:
        entities=logical_rows//samples
        ds=group.create_dataset("values",shape=(rows,entities,samples),maxshape=(None,entities,samples) if resizable else None,dtype="uint64",chunks=(1,min(entities,step),1),compression=compression,shuffle=True,fletcher32=True)
        ds.attrs["codec"]=NATIVE_STORED_CODEC
        for name in ("samples_per_entity","entity_count","logical_rows","association_sha256"):
            ds.attrs[name]=response.attrs[name]
    return ds


def plain_stored_dataset(group,values,compression):
    values=np.asarray(values)
    options=dict(chunks=(1,min(values.shape[1],CHUNK_BYTES//8)),compression=compression,shuffle=True,fletcher32=True) if values.size else {}
    ds=group.create_dataset("values",shape=values.shape,dtype="float64",**options)
    if values.size:
        for row in range(values.shape[0]):
            for start in range(0,values.shape[1],ds.chunks[1]):
                stop=min(start+ds.chunks[1],values.shape[1])
                ds[row,start:stop]=values[row,start:stop]
    return ds


def logical_stored_shape(dataset):
    if dataset.attrs.get("codec") in (NATIVE_STORED_CODEC,NATIVE_STORED_CODEC.encode()):
        return (dataset.shape[0],dataset.shape[1]*dataset.shape[2])
    return dataset.shape


def encode_tile_rows(width):
    # At most two source/decoded basis matrices plus selector/scatter, predictor,
    # conversion, mask buffers and a coefficient row. No writer response cache.
    if width*48+128 > ENCODE_WORK_BYTES:
        raise WorkingSetError("one stored predictor row exceeds bounded encoder working capacity")
    return max(1, min(CHUNK_BYTES//8, (ENCODE_WORK_BYTES-width*32)//(width*16+128)))


def read_basis_tile(response, start, stop):
    if response.ndim == 2:
        return np.asarray(response[start:stop], dtype=np.float64)
    samples = int(response.attrs["samples_per_entity"])
    output = np.empty((stop-start, response.shape[2]), dtype=np.float64)
    for station in range(samples):
        first = start + (station-start) % samples
        if first >= stop:continue
        last = stop-1
        source = response[first//samples:last//samples+1:1, station, :]
        positions = np.arange(first-start, stop-start, samples)
        output[positions] = source[:len(positions)]
    return output


def write_corrections(dataset, actual_values, case_indices, response, coefficients, cases, *, start_row=0, on_write=None, on_work=None):
    width=response.shape[-1]
    native=dataset.attrs.get("codec") in (NATIVE_STORED_CODEC,NATIVE_STORED_CODEC.encode())
    samples=response.shape[1] if native else 1
    rows=response.shape[0] if native else actual_values.shape[1]
    # Align work tiles to the persisted entity/logical-row chunk. Each complete
    # case/spatial/station chunk is compressed once, including batch append.
    step=dataset.chunks[1]
    for first in range(0,rows,step):
        stop=min(first+step,rows)
        for sample in range(samples):
            basis=np.asarray(response[first:stop,sample,:],dtype=np.float64) if native else read_basis_tile(response,first,stop)
            if not np.isfinite(basis).all():raise ValidationError("stored predictor source contains nonfinite values")
            if on_work is not None:on_work((stop-first)*(width*16+128)+width*32)
            for row,case_index in enumerate(case_indices):
                ci=int(case_index)
                weights=np.asarray(coefficients[ci],dtype=np.float64)
                if not np.isfinite(weights).all():raise ValidationError("stored predictor coefficients contain nonfinite values")
                bits=predictor_bits(basis,weights,cases[ci].get("active_indices"))
                selector=slice(first*samples+sample,stop*samples+sample,samples) if native else slice(first,stop)
                actual=np.asarray(actual_values[row,selector],dtype=np.float64)
                mask=np.bitwise_xor(actual.view(np.uint64),bits)
                if native:dataset[start_row+row,first:stop,sample]=mask
                else:dataset[start_row+row,first:stop]=mask
                if on_write is not None:on_write(mask.nbytes)
