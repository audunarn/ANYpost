"""Reproducible archive/retained GPU performance check (synthetic, no physics claim)."""
from __future__ import annotations
import argparse, json, os, platform, sys, time, hashlib
from dataclasses import asdict
from pathlib import Path
import importlib.metadata
import numpy as np
import psutil


def fixture(derived=False, samples_per_element=1):
    nx,ny=250,200
    x,y=np.meshgrid(np.arange(nx+1,dtype=float),np.arange(ny+1,dtype=float))
    positions=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
    lower=(np.arange(ny)[:,None]*(nx+1)+np.arange(nx)[None,:]).ravel()
    triangles=np.empty((2*len(lower),3),np.uint32)
    triangles[::2]=np.column_stack((lower,lower+1,lower+nx+1))
    triangles[1::2]=np.column_stack((lower+1,lower+nx+2,lower+nx+1))
    count=len(triangles)
    entity_ids=np.arange(1,count+1,dtype=np.int64)
    association_ids=np.repeat(entity_ids,samples_per_element)
    locations=count*samples_per_element
    # Match current producer metadata as well as numeric row counts. The same
    # source table is shared in memory; archive JSON preserves each descriptor.
    native_shapes={str(int(entity)):[samples_per_element] for entity in entity_ids}
    rng=np.random.default_rng(20261012)
    coefficients=rng.uniform(-2,2,(10000,20))
    responses={name:rng.normal(0,1e5,(locations,21)) for name in ("stress/xx","stress/yy")}
    derived_specs={}
    if derived:
        names=("membrane_xx","membrane_yy","membrane_xy","bending_xx","bending_yy","bending_xy","shear_xz","shear_yz")
        for name in names: responses[f"stress/{name}"]=rng.normal(0,1e5,(locations,21))
        inputs={name:f"stress/{name}" for name in names}
        derived_specs={"stress/shell_von_mises":dict(operation="shell_von_mises",inputs=inputs),
                       "stress/principal_top_1":dict(operation="shell_principal_stress",inputs=inputs,surface="top",rank=1)}
    return dict(schema="anysolver.recovery_capsule",version=1,runtime="anysolver-results/1",
                model_identity="synthetic-switching-grid-100000-v1",mesh_identity="synthetic-triangle-grid-250x200-v1",
                solver_identity={"kind":"synthetic-performance-fixture","physics_claim":False,"seed":20261012},
                geometry=dict(positions=positions,triangles=triangles,node_ids=np.arange(1,len(positions)+1,dtype=np.int64),
                              element_ids=entity_ids,triangle_to_element=np.arange(count,dtype=np.uint32)),
                patterns=tuple(f"p{i}" for i in range(20)),coefficients=coefficients,
                cases=tuple(dict(case_id=f"c{i}",status="converged",certified=True,route="synthetic-runtime-check",
                                 active_indices=list(range(1,21))) for i in range(10000)),
                descriptors=tuple(dict(key=k,quantity_id="stress",component=k.split("/")[1],location="element",
                                       unit="Pa",basis="local",sampling="native_samples",availability="reconstructable",
                                       metadata={"samples_per_element":samples_per_element,"element_shapes":native_shapes}) for k in (*responses,*derived_specs)),
                associations={k:association_ids for k in (*responses,*derived_specs)},responses=responses,stored={},derived=derived_specs)


def summary(records):
    latency=np.array([r["latency_ms"] for r in records])
    return dict(count=len(records),p50_ms=float(np.percentile(latency,50)),p95_ms=float(np.percentile(latency,95)),
                maximum_ms=float(latency.max()),
                stages_p95_ms={name:float(np.percentile([(r[b]-r[a])/1e6 for r in records],95))
                               for name,a,b in [("read_and_reconstruct","requested_ns","loaded_ns"),
                                                ("dispatch_and_apply","loaded_ns","applied_ns"),
                                                ("render_and_swap","applied_ns","presented_ns")]})


def package_identities():
    """Record exact loaded source as well as distribution metadata."""
    import anypost, anysolver_results, anyfem, any3dview
    result = {}
    for name, module in (("ANYpost", anypost), ("ANYsolver-results", anysolver_results),
                         ("ANYfem", anyfem), ("ANY3dView", any3dview)):
        try: distribution_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: distribution_version = None
        directory = Path(module.__file__).resolve().parent
        files = {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(directory.rglob("*.py"))}
        result[name] = dict(distribution_version=distribution_version,
                            module_version=getattr(module, "__version__", None),
                            module_path=str(module.__file__), source_sha256=files)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--requests",type=int,default=100)
    parser.add_argument("--derived",action="store_true")
    parser.add_argument("--samples-per-element",type=int,choices=(1,4,7),default=1)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    from anypost import write_results
    started=time.perf_counter()
    capsule=fixture(args.derived,args.samples_per_element)
    generation_seconds=time.perf_counter()-started
    archive=args.output/"100k-10k-20patterns.anypost.h5"
    started=time.perf_counter()
    write_results(archive,capsule)
    export_seconds=time.perf_counter()-started
    explicit_field_bytes=100000*args.samples_per_element*10000*2*8
    basis_bytes=sum(a.nbytes for a in capsule["responses"].values())
    del capsule
    import tkinter as tk
    from any3dview.gpu.widget import Any3DView
    from anyfem.post.archive import ArchiveSession
    from threadpoolctl import threadpool_info
    from concurrent.futures import ThreadPoolExecutor
    started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix="anypost-open") as executor:
        prepared=executor.submit(ArchiveSession.prepare,archive).result()
    prefetch_seconds=time.perf_counter()-started
    root=tk.Tk()
    root.title("ANYpost — 100,000-element verification")
    root.geometry("900x700")
    viewer=Any3DView(root,width=900,height=700)
    viewer.pack(fill="both",expand=True)
    category_names=(("resident_derived_field","resident_von_mises_basis","resident_principal_basis","uncached_von_mises","uncached_principal")
                    if args.derived else ("resident_field","resident_basis","uncached_archive"))
    records={k:[] for k in category_names}
    state=dict(mode=0,index=0,pending=None,started_ns=None,error=None)
    modes=list(records)
    session=None
    rss_peak=psutil.Process().memory_info().rss
    initial_generations=None
    mode_diagnostics={}
    baseline_read_count=None

    def begin_mode():
        nonlocal baseline_read_count
        mode=modes[state["mode"]]
        state["index"]=0
        session.reader.clear_cache()
        def warm(component):
            selection=(session.sample_selection("stress",component,1)
                       if args.samples_per_element>1 else None)
            session.reader.read_field("c0","stress",component,selection=selection)
        # Prepare actual warm inputs before measured viewer requests.
        if mode=="resident_field":
            for component in ("xx","yy"):
                warm(component)
        elif mode=="resident_basis":
            warm("xx")
        elif mode=="resident_derived_field":
            for component in ("shell_von_mises","principal_top_1"):
                warm(component)
        elif mode=="resident_von_mises_basis":
            warm("shell_von_mises")
        elif mode=="resident_principal_basis":
            warm("principal_top_1")
        baseline_read_count=session.reader.diagnostics["reads"]
        root.after(1,request)

    def request():
        mode=modes[state["mode"]]
        i=state["index"]
        if mode.startswith("uncached_"):
            # Previous request has presented and retired before this callback.
            # Clear application native selection/conversion caches as well as
            # archive numeric caches; retained geometry remains the same.
            session._clear_sample_map()
            session.reader.clear_cache()
        case="c0" if mode in ("resident_field","resident_derived_field") else f"c{i+1+state['mode']*100}"
        component=(("shell_von_mises","principal_top_1")[i%2] if mode=="resident_derived_field" else
                   "shell_von_mises" if "von_mises" in mode else "principal_top_1" if "principal" in mode else
                   ("xx","yy")[i%2] if mode=="resident_field" else "xx")
        state["started_ns"]=time.perf_counter_ns()
        state["pending"]=(session.request_native_sample(case,"stress",component,1)
                          if args.samples_per_element>1 else session.request_component(case,"stress",component))

    def presented(record):
        nonlocal rss_peak
        if state["pending"] is None or record.request_id!=state["pending"]: return
        mode=modes[state["mode"]]
        item=asdict(record)
        item["latency_ms"]=record.latency_ms
        records[mode].append(item)
        if len(records[mode])%10==0:
            (args.output/"progress.json").write_text(json.dumps({"mode":mode,"records":records}),encoding="utf8")
        rss_peak=max(rss_peak,psutil.Process().memory_info().rss)
        state["index"]+=1
        if state["index"]<args.requests:
            root.after(1,request)
        else:
            mode_diagnostics[mode]=dict(session.reader.diagnostics)
            mode_diagnostics[mode]["reads_during_mode"]=session.reader.diagnostics["reads"]-baseline_read_count
            state["mode"]+=1
            if state["mode"]==len(modes): root.after(1,finish)
            else: root.after(1,begin_mode)

    def watchdog():
        if state["error"]: return
        if session.view.error is not None:
            state["error"]=repr(session.view.error)
            finish()
            return
        if state["started_ns"] and (time.perf_counter_ns()-state["started_ns"])/1e9>20:
            state["error"]="one request exceeded the 20-second timeout"
            finish()
            return
        root.after(20,watchdog)

    def finish():
        if state.get("finished"): return
        state["finished"]=True
        final_generations=asdict(session.handle.generations)
        memory_info = psutil.Process().memory_info()
        true_peak = getattr(memory_info, "peak_wset", None)
        result=dict(fixture="synthetic performance geometry and scalar basis; no solver physics qualification",
                    elements=100000,cases=10000,patterns=20,geometry_nodes=50451,
                    native_samples_per_element=args.samples_per_element,
                    archive_locations=100000*args.samples_per_element,
                    displayed_native_sample=1 if args.samples_per_element>1 else None,
                    generation_seconds=generation_seconds,archive_export_seconds=export_seconds,
                    archive_prefetch_seconds=prefetch_seconds,
                    archive_bytes=archive.stat().st_size,primitive_response_bytes=basis_bytes,
                    explicit_two_field_numeric_bytes=explicit_field_bytes,derived_operations=args.derived,
                    note="Explicit numeric byte count is exact arithmetic, not a measured explicit archive size.",
                    os_cache_state="Not flushed. OS cache may be warm; uncached mode clears archive numeric and native-sample application caches before each timed request; retained geometry/display mappings remain.",
                    native_sample_diagnostics=dict(session.native_sample_diagnostics),
                    cache_modes=mode_diagnostics,process_rss_peak_bytes=true_peak if true_peak is not None else rss_peak,
                    process_rss_sampled_peak_bytes=rss_peak,
                    peak_rss_method="Windows peak working set" if true_peak is not None else "sampled resident set",
                    machine=dict(platform=platform.platform(),processor=platform.processor(),
                                 logical_cpus=psutil.cpu_count(),physical_cpus=psutil.cpu_count(logical=False),
                                 memory_bytes=psutil.virtual_memory().total,gpu=dict(viewer._renderer.ctx.info)),
                    numpy_threadpools=threadpool_info(),python=sys.version,
                    package_versions={p:importlib.metadata.version(p) for p in ("numpy","h5py","moderngl","ANY3dView","ANYfem")},
                    package_identities=package_identities(),
                    initial_generations=initial_generations,final_generations=final_generations,
                    measurements={k:summary(v) for k,v in records.items() if v},records=records,error=state["error"])
        result["correctness_checks"]=dict(topology_unchanged=initial_generations["topology"]==final_generations["topology"],
                                          position_unchanged=initial_generations["position"]==final_generations["position"],
                                          deformation_unchanged=initial_generations["displacement"]==final_generations["displacement"],
                                          cache_budget_respected=all(d["cache_resident_bytes"]<=d["cache_bytes"] for d in mode_diagnostics.values()))
        result["targets_passed"]=(state["error"] is None and all(len(records[k])>=100 for k in modes)
                                  and all(summary(records[k])["p95_ms"]<(1000 if k.startswith("uncached_") else 100) for k in modes)
                                  and all(result["correctness_checks"].values()))
        (args.output/"result.json").write_text(json.dumps(result,indent=2),encoding="utf8")
        session.close()
        root.destroy()

    try:
        session=ArchiveSession(prepared,viewer,on_presented=presented)
        viewer.fit_to_scene()
        initial_generations=asdict(session.handle.generations)
        root.after(250,begin_mode)
        root.after(500,watchdog)
        root.mainloop()
    finally:
        if session is not None: session.close()
        else: prepared.close()
        try: root.destroy()
        except tk.TclError: pass
    data=json.loads((args.output/"result.json").read_text(encoding="utf8"))
    print(json.dumps({"targets_passed":data["targets_passed"],"measurements":data["measurements"],
                      "archive_bytes":data["archive_bytes"],"result":str(args.output/"result.json")}))
    return 0 if data["targets_passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
