"""Optional lossless response layout; association grouping conveys no physics."""
import hashlib
import numpy as np
from .errors import ValidationError

CODEC = "uniform_samples/1"
CHUNK_BYTES = 256 * 1024


def association_digest(values, block_rows=4096):
    digest = hashlib.sha256()
    for start in range(0, len(values), block_rows):
        block = np.asarray(values[start:start+block_rows], dtype="<i8")
        digest.update(block.tobytes(order="C"))
    return digest.hexdigest()


def uniform_samples(associations):
    if associations is None:
        return None
    a = np.asarray(associations)
    if a.ndim not in (1, 2) or (a.ndim == 2 and not a.shape[1]) or len(a) < 2:
        return None
    leading = a if a.ndim == 1 else a[:, 0]
    # Only the association leading ID is a storage hint. Preserve all columns.
    changes = np.flatnonzero(leading[1:] != leading[:-1]) + 1
    samples = int(changes[0]) if len(changes) else len(a)
    if samples <= 1 or len(a) % samples:
        return None
    if len(changes) != len(a)//samples - 1 or not np.array_equal(changes, np.arange(samples, len(a), samples)):
        return None
    group_ids = leading[::samples]
    if len(np.unique(group_ids)) != len(group_ids):
        return None
    return samples


def response_dataset(group, values, associations, compression, flat_writer, on_write=None):
    # Dataset sources remain sliceable: never reshape/copy a complete F-order basis.
    width = values.shape[1]
    samples = uniform_samples(associations)
    if samples is None or width * 8 > CHUNK_BYTES:
        return flat_writer(group, "responses", values, compression)
    entities = values.shape[0] // samples
    step = max(1, min(entities, CHUNK_BYTES // (width * 8)))
    ds = group.create_dataset("responses", shape=(entities, samples, width), dtype="float64",
                              chunks=(step, 1, width), compression=compression,
                              shuffle=True, fletcher32=True)
    ds.attrs.update(codec=CODEC, samples_per_entity=samples, entity_count=entities,
                    logical_rows=values.shape[0], association_sha256=association_digest(associations))
    for sample in range(samples):
        for start in range(0, entities, step):
            stop = min(start+step, entities)
            block = np.asarray(values[start*samples+sample:stop*samples+sample:samples], dtype="float64")
            ds[start:stop, sample, :] = block
            if on_write is not None:
                on_write(block.nbytes)
    return ds


def response_shape(ds, logical_rows, width):
    codec = ds.attrs.get("codec")
    if codec is None:
        if ds.ndim != 2 or ds.shape != (logical_rows, width):
            raise ValidationError("invalid response dataset")
        return None
    if isinstance(codec, bytes):
        codec = codec.decode("utf-8")
    if codec != CODEC:
        raise ValidationError("unknown response codec")
    def integer(name):
        value = ds.attrs.get(name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValidationError("invalid uniform sample codec metadata")
        return int(value)
    samples, entities, count = (integer(name) for name in ("samples_per_entity", "entity_count", "logical_rows"))
    digest = ds.attrs.get("association_sha256", "")
    if isinstance(digest, bytes): digest = digest.decode("utf-8")
    if samples <= 1 or entities <= 0 or count != logical_rows or entities*samples != count or ds.shape != (entities, samples, width):
        raise ValidationError("uniform sample codec shape mismatch")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValidationError("invalid uniform sample association binding")
    return samples
