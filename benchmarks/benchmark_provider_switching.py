"""Cold/warm provider timing using an explicit native numeric fixture.

An optional .npy is a native spatial-first result array, never an object archive.
The default is synthetic; these timings do not measure presentation or physics.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import numpy as np
import anypost.provider as current


def run(module, values, repeats):
    class Provider:
        def metadata(self):
            return dict(model_identity='explicit-provider-benchmark',
                cases=[dict(id='0', status='completed')],
                descriptors=[dict(quantity_id='values', components=('value',),
                    spatial_count=len(values), location='element',
                    value_shape=values.shape[1:], provider_work_bytes=0)])
        def read_field(self, *args, selection=None, cancel_event=None):
            return values[selection]
    samples={mode: [] for mode in ('cold', 'warm')}
    with module.open_memory_results(Provider()) as reader:
        for _ in range(repeats):
            for mode in samples:
                if mode == 'cold': reader.clear_cache()
                start=perf_counter()
                result=reader.read_field('0', 'values', 'value', reduction='mean')
                samples[mode].append((perf_counter()-start)*1000)
        return dict(timing_ms={mode:dict(count=len(times),
                median=float(np.median(times)), p95=float(np.percentile(times,95)))
                for mode,times in samples.items()},
                result_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
                diagnostics=reader.diagnostics)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--values', type=Path)
    parser.add_argument('--repeats', type=int, default=10)
    args=parser.parse_args()
    if args.repeats<1: parser.error('--repeats must be positive')
    values=(np.load(args.values,allow_pickle=False) if args.values else
            np.random.default_rng(20261020).normal(size=(27111,3)))
    spec=importlib.util.spec_from_file_location('anypost._provider_baseline',args.baseline)
    baseline=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
    report=dict(python=sys.executable, version=platform.python_version(),
        numpy=np.__version__, source=current.__file__, baseline=str(args.baseline),
        source_sha256=hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
        baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        thread_settings={name:os.environ.get(name) for name in
                         ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        input_kind='explicit .npy' if args.values else 'synthetic',
        input=str(args.values) if args.values else None,
        input_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
        shape=values.shape, dtype=str(values.dtype),
        before=run(baseline,values,args.repeats), after=run(current,values,args.repeats))
    if report['before']['result_sha256']!=report['after']['result_sha256']:
        raise AssertionError('provider results changed')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
