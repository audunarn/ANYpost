"""Matched measured storage comparison; bounded 100-case sample, not extrapolated evidence."""
import argparse,json,time
from pathlib import Path
import numpy as np
from benchmark_switching import fixture
from anypost import write_results,open_results,ArchiveSink

parser=argparse.ArgumentParser()
parser.add_argument("--source",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--samples-per-element",type=int,choices=(1,4,7),default=1)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
capsule=fixture(samples_per_element=args.samples_per_element)
capsule["cases"]=capsule["cases"][:100]
capsule["coefficients"]=capsule["coefficients"][:100]
started=time.perf_counter()
compact=write_results(args.output/"compact-100.h5",capsule)
compact_seconds=time.perf_counter()-started
explicit=args.output/"explicit-100.h5"
descriptors=tuple(dict(d,availability="stored") for d in capsule["descriptors"])
header={k:v for k,v in capsule.items() if k not in ("cases","coefficients","stored","descriptors","responses")}
header["responses"]={}
materialisation_seconds=0.0
largest_live_case_field_bytes=0
started=time.perf_counter()
with open_results(args.source,cache_bytes=256*1024**2) as reader,ArchiveSink(explicit) as sink:
    sink.capsule_begin(header)
    for i,case in enumerate(capsule["cases"]):
        sink.capsule_case(case,capsule["coefficients"][i])
    # A 256 MiB cache retains one full native basis, not both. Process one
    # quantity at a time rather than rereading each basis for every case.
    for descriptor in descriptors:
        for i in range(len(capsule["cases"])):
            reading=time.perf_counter()
            values=reader.read_field(f"c{i}","stress",descriptor["component"])
            materialisation_seconds+=time.perf_counter()-reading
            largest_live_case_field_bytes=max(largest_live_case_field_bytes,values.nbytes)
            sink.capsule_stored(descriptor["key"],np.array([i],dtype=np.int64),values[None,:])
            del values
    sink.capsule_end(dict(descriptors=descriptors,associations=capsule["associations"],responses={},derived={},case_count=100,stored_superseded_cases={}))
explicit_total_seconds=time.perf_counter()-started
explicit_seconds=explicit_total_seconds-materialisation_seconds
with open_results(compact) as a,open_results(explicit) as b:
    for i in (0,1,37,99):
        for component in ("xx","yy"):
            np.testing.assert_array_equal(a.read_field(f"c{i}","stress",component),b.read_field(f"c{i}","stress",component))
result=dict(fixture="same synthetic 100000 elements,100 cases,20 patterns,two primitive fields",
            samples_per_element=args.samples_per_element,spatial_values_per_field=100000*args.samples_per_element,
            compact_archive_bytes=compact.stat().st_size,explicit_archive_bytes=explicit.stat().st_size,
            measured_size_ratio=explicit.stat().st_size/compact.stat().st_size,
            compact_write_seconds=compact_seconds,explicit_materialisation_seconds=materialisation_seconds,
            explicit_write_seconds=explicit_seconds,bitwise_sample_checks=8,
            explicit_read_and_write_seconds=explicit_total_seconds,largest_live_case_field_bytes=largest_live_case_field_bytes,
            sink_diagnostics=sink.diagnostics,
            note="Measured 100-case comparison only. 10000-case explicit storage was not written or extrapolated as measured savings.")
(args.output/"result.json").write_text(json.dumps(result,indent=2),encoding="utf8")
print(json.dumps(result))
