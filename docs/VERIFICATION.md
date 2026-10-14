# ANYpost V1 verification — current evidence, 2026-10-13

The coordinated candidate passes the synthetic GPU targets and focused platform checks. Acceptance remains incomplete: the physical 100,000-element, 10,000-case, 20-pattern archive and measured physical storage comparison are unexecuted. No release is accepted or published. The two low-severity review repairs are frozen in candidate 004; its final platform binding remains pending. The physical 10,000-element readiness attempt failed during preparation, and the newly verified memory repair is not yet installed in the main environment.

## Coordinated candidate 003

Frozen identities are in [wheels.json](C:/ANYworkspaceWork/anypost-20261012/coordinated-runtime2-candidate-003/wheels.json): ANYsolver 6FBA41CD, ANYsolver-results D0285773, ANYpost 1E4478D4, ANYfem B2168F60 and ANY3dView 360EAF23. The manifest records full SHA-256 values. Clean Windows installation passed in 12.884312 seconds; installed integration passed 55 tests in 11.880401 seconds with all Python package members bound to their wheels.

GPU check 004 completed 500 actual presented-frame requests in 99.948258 seconds. Each mode contains 100 requests. Uncached modes clear application caches; operating-system caches were not reset. The fixture contains synthetic 100,000-element geometry, 10,000 cases and 20 patterns. It establishes presentation performance on this fixture, not physical source scale correctness or compactness.

| Mode | Requests | p50 ms | p95 ms |
|---|---:|---:|---:|
| resident_derived_field | 100 | 14.32 | 15.53 |
| resident_von_mises_basis | 100 | 30.27 | 46.12 |
| resident_principal_basis | 100 | 77.23 | 93.40 |
| uncached_von_mises | 100 | 348.99 | 383.33 |
| uncached_principal | 100 | 406.28 | 452.81 |

All p95 targets passed. Topology, positions, deformation, retained geometry arrays and the cache ceiling checks passed. Peak process RSS was 1,010,069,504 bytes. [Raw timing evidence](C:/ANYworkspaceWork/anypost-20261012/coordinated-runtime2-500-gpu-output-004/result.json) includes stage timings and actual package, hardware and effective thread settings. Prior 108.04 ms and 106.77 ms principal failures remain preserved.

Linux repaired-wheel verification passed 260 pure tests and 35 narrow adapter tests, with one explicitly deselected Windows source-producer test. The unchanged original Windows golden files produced 430 exact fields, 2,150 uint64 assertions, 7,430 output words, 86 envelope identities, 196 association checks and 60 typed unavailable refusals. An actual installed reader matched 15,009 serial principal words. Core/private package members (207 Python files) match the wheels; neither fullsolver nor SciPy is installed in the core environment. The adapter uses a narrow fixture because ANYgeometry is absent, so full Linux application support is not claimed. [Linux handoff](C:/ANYworkspaceWork/anypost-20261012/Linux-repair3-evidence/HANDOFF.json) records the complete binding.

Source verification passed 108 tests and an installed-wheel smoke check with 60 exact comparisons on three cases. At 1,000 physical elements, 20 independent patterns and 20 completed cases, existing 16-case batches took 0.629358 seconds for automatic export and 8.027134 seconds for preparation. Complete capsules matched the original record route and earlier reference, with all 20,000 original per-input element GEMVs retained. These measurements are not extrapolated acceptance for the large model. [Source handoff](C:/ANYworkspaceWork/anypost-20261012/source-scale-next-evidence/HANDOFF.json) preserves stage counts, original inputs and the failed comparator-harness attempt.

Independent Opus 5.5/high repair review completed in 273.410899 seconds. Its read-only tool scope audit is clean. It found no bypass of the six repaired classes and no wrong successful result. Two low residual findings concerned finite eigenvalue/error handling and same-object snapshot byte growth; both now have focused source and private installed-wheel checks. A subsequent narrow Opus review found no repair defect, but its filename Glob exceeded the assigned search scope; that deviation is preserved and the scope audit is not labelled clean. Informational observations and exclusions remain in the [review](C:/ANYworkspaceWork/anypost-20261012/claude-repair-audit-001/REPORT.md). This review is not final acceptance.

## Current preparation failure and repair

Candidate 004 installs source9EB7139A, runtime8A5FB8E and the unchanged consumer wheels. The runtime finite-output repair passed 55 source and 55 private installed checks; the snapshot accounting repair passed three source and three installed checks. [Candidate bindings](C:/ANYworkspaceWork/anypost-20261012/coordinated-runtime2-candidate-004/wheels.json) and [narrow review](C:/ANYworkspaceWork/anypost-20261012/claude-low-delta-audit-001/REPORT.md) preserve their exact identities and limits.

The actual 10,000-element, 20-pattern, 16-case attempt failed during preparation after 137.826064 seconds. No result archive was exported. Its recovery-affinity exception lacked detailed diagnostics. The RSS monitor observed 17,378,480,128 bytes before its stop attempt, but the original process-tree termination failed; this is an observed value, not a proven lifetime peak. [Original attempt](C:/ANYworkspaceWork/anypost-20261012/source-10000-readiness/attempt-1791845569842663600/process.json) remains preserved. The repaired termination harness was verified using disposable sleeping processes.

Source3F2153C0 removes a dense zero rigid-mode grid for models whose components are all supported, preserving mixed/free calculations and diagnostics. Fifteen source and fifteen private installed checks passed; only assembly.py differs from source9EB. A separate original-grid element4808 probe reproduced the public axial rigid-offset refusal (6.355546660296293e-9 versus the unchanged 1e-9 tolerance). This is new small-model evidence, not a recovered diagnostic from the earlier large failure. [Memory handoff](C:/ANYworkspaceWork/anypost-20261012/source-supported-memory-evidence/HANDOFF.md) and [affinity evidence](C:/ANYworkspaceWork/anypost-20261012/affinity-small-discriminator/REPORT.md) describe the checks.

A separate explicit opt-in native-element fallback is being implemented. The default library refusal and original scientific gates remain in place. Rejected elements must have complete actual source outputs rather than an advertised qualified basis. No 10,000-element retry or 100,000-element execution is currently authorised by the consumed readiness run; fresh controls and final artifact binding remain required.

## Historical evidence

The following measurements and checks bind earlier candidates. Their earlier timings and package identities do not establish acceptance of the current candidate.

## Measured Windows switching

The fixture has 100,000 triangular elements, 50,451 nodes, seven native samples per element (700,000 logical locations), 10,000 retained cases and 20 response patterns. It includes eight primitive shell stress dependencies, two additional scalar fields, nonlinear von Mises/principal measures and the producer's 100,000-entry per-component shape metadata. Response data are synthetic; physical recovery correctness is independently checked with authoritative small beam/shell fixtures.

Each row contains 100 requests timed from request to successful Tk swap. Warm basis modes change case IDs; cached fields toggle components. Cold modes clear both application native-sample maps and ANYpost numeric caches before every request. Geometry, entity-to-render maps and immutable metadata/admission proofs remain retained. OS caches were not flushed; these are application-cache misses, not proven cold-disk measurements.

| Mode | Requests | p50 ms | p95 ms | Max ms |
|---|---:|---:|---:|---:|
| resident_derived_field | 100 | 29.19 | 30.47 | 31.23 |
| resident_von_mises_basis | 100 | 30.06 | 45.77 | 46.51 |
| resident_principal_basis | 100 | 78.08 | 93.33 | 108.26 |
| uncached_von_mises | 100 | 325.18 | 375.37 | 398.62 |
| uncached_principal | 100 | 361.88 | 409.68 | 449.05 |

All warm p95 values are below 100 ms; both uncached p95 values are below 1,000 ms. Principal maximum latency was above 100 ms; the agreed gate is p95. Topology, positions and deformation generations stayed unchanged.

Windows 11, AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD, 16 physical/32 logical CPUs, 127.1 GiB system RAM; NVIDIA GeForce RTX 3090 Ti, OpenGL 4.1. Python 3.11.9; NumPy 2.4.6, h5py 3.16.0. Both effective OpenBLAS pools reported one thread. Package paths and every loaded product Python-file hash are in the raw timing evidence.

The complete archive occupies 1,162,433,216 bytes. Fixture preparation took 1.630 s; archive export 5.824 s; background header/geometry preparation 3.176 s. Total bounded process wall was 102.359579 s. True Windows peak process working set was 1,308,413,952 bytes (includes fixture creation). Managed cache peak was 396,400,160 bytes under the 2 GiB ceiling. Archive-managed transient peak was 33,600,000 bytes; native-map preparation separately checks its conservative 56,000,256-byte bound against the same 256 MiB working cap. Python/HDF5/driver overhead and geometry are separately visible in process RSS.

## Measured storage comparison

Matched 100 cases, 100,000 elements � seven samples, 20 patterns and two primitive fields: compact 231,504,756 bytes; explicit 1,092,467,906 bytes, a measured 4.719� size ratio. Eight complete case/component comparisons matched bitwise. Explicit output streamed one 5.6 MB field at a time, with 256 KiB write blocks and zero retained explicit value bytes. It took 13.880 s to reconstruct and 6.739 s for writing/finalisation. No 10,000-case explicit archive was written; no saving at that scale is claimed.

The sample-separated codec reduced a separate native cache-cold read from 1.667 s to 0.327 s. On that highly repetitive fixture it increased archive size from 229 MB to 1.151 GB. Compression benefit depends on data; flat native/legacy readers and uncompressed writing remain supported.

The first final native storage comparison hit its 120 s limit because a 256 MiB cache could not retain both full native bases while requests interleaved components. Its partial temporary archive and diagnostics remain preserved; the termination harness failed to record exact total wall time, so that duration is unknown. Processing one component at a time repaired the measured cache thrashing; the successful comparison took 34.127945 s with the same 120 s bound. No incomplete archive was published.

## Correctness and installation

- All 106 installed ANYpost tests pass: 10 passed initially, 94 after a system-temporary-directory repair, and two remaining producer tests after correcting Windows test-fixture path quoting. The initial 96 setup errors and two missing-fixture failures remain preserved. Product packages imported from the clean environment; only test helpers came from source.
- Installed exporter-to-archive round trips made 258 field comparisons across certified beam, pressure-dependent certified shell and canonical shell cases. They cover baseline offsets, displacements, reactions, primitive/derived values, reverse/duplicate selections, shared recovery metadata and source-index envelope identities. Float64 data stay read-only. Existing scientific tolerance remained 1e-12 where applicable; displacement/reaction checks are bitwise.
- Nine installed source-owned frame/sampling tests pass, including the admitted imperfect-midpoint quadratic-beam endpoint frame, ordinary section envelopes versus generalized beam station rules, rotated S3 frames, model mutation independence and canonical/fallback metadata.
- Twenty-one consumer tests cover worker preparation, owner attachment, vector refusal, cached scalar-shape proof reuse, cancellation, stale completions, mapping, budget refusal and cleanup. Final Tk and Qt installed gates pass in 6.374533 and 5.116580 s. They verify scalar-only uploads, distinct endpoint pixels, native NaN subsets, latest selection and owning-thread callbacks. Startup archive I/O runs on a worker. Qt uses its external frameSwapped boundary; its native PresentedFrame callback remains unsupported and is explicitly refused.
- Clean isolated Windows/Python 3.11 and Linux/Ubuntu 24.04 Python 3.12 headless smoke checks pass. Linux uses NumPy 2.4.3/h5py 3.15.1; no full solver, ANYfem or SciPy modules load in headless reconstruction. Windows pip dependency checking passes.
- A streamed envelope over all 10,000 cases at three selected native locations matches an independent flat-source oracle exactly, including governing IDs; 2 MiB cache/work caps held, 73 MB process peak, 7.100 s envelope. This is not a full-model envelope timing claim. All 147 million native fixture response values and source associations matched exactly in the codec conversion check.

The exporter keeps explicit per-case rows when source arithmetic, load-dependent recovery or canonical/fallback routes cannot certify portable reconstruction. Derived measures are recomputed from primitive fields using the shared solver-owned formulas. The runtime never certifies new combinations. Shared optional model metadata is preserved as finite JSON; downstream coordinate transformations would require their own schema admission, rather than treating arbitrary metadata as physics certification.

The final ANYfem wheel supersedes the tested 15D472 wheel only in dependency metadata: `ANYfem[post]` requests `ANYpost[results]`, guaranteeing the lightweight runtime even with an older installed solver. All 156 Python members are byte-identical to the tested GUI/performance artifact; no numerical or rendering code changed. The final source/wheel SHA256 audit and dependency checks bind this metadata-only artifact.

## Artifacts

See [manifest.json](C:/ANYworkspaceWork/anypost-20261012/delivery/manifest.json) for wheel SHA256 identities, source-byte audits and full evidence paths. Windows installation with Python 3.11+:

```powershell
./install.ps1 -Python "path/to/python.exe"
```

This installs explicit local wheels and their normal pip dependencies. The ANYloads wheel is included because version 0.1.0 was unavailable from the package index. Its source checkout was not edited. Run `anyfem-archive analysis.anypost.h5`; Python integrations use `ArchiveSession.prepare` on their I/O executor, then `ArchiveSession(prepared, viewer)` on the viewer owner. Linux headless consumers need only the ANYpost and ANYsolver-results wheels plus their NumPy/h5py dependencies.

All changes remain local and uncommitted. Existing unrelated dirty work is preserved. ANYmarine, transient/nonlinear recovery, new physical combinations and publication remain outside this V1 delivery.

The Windows installer explicitly reinstalls these local wheels before resolving normal dependencies, so previously installed packages with the same version cannot shadow the new code. It includes the renderer GPU/Qt extras and ANYtk3D. The measured latency configuration uses one BLAS thread; configure the process before Python imports NumPy:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
anyfem-archive analysis.anypost.h5
```

For benchmarks, additionally install `psutil` and `threadpoolctl`. The reusable libraries do not alter the host application's global thread settings.


## Completion-audit candidate amendment

Current candidate and requirement matrix: [manifest.json](C:/ANYworkspaceWork/anypost-20261012/audit-candidate-001/manifest.json).
The loader releases completed/superseded/closed field buffers and failed UI
dispatch references;13 impacted installed checks pass. The exporter preserves
failed-record diagnostics without certifying failed fields;10 installed exporter
checks pass. All911 Python wheel members match current source. Refreshed Tk,
Qt and Linux checks pass. The current500-request synthetic GPU run has warm
p95 values30.72/45.93/93.04ms and uncached363.45/420.37ms.

A two-element physical shell export retained10000 certified20-pattern cases,
but8847-9075 cases needed stored overrides for several quantities to preserve
source rounding. Real100k producer/export compactness remains unproven. A
storage-only XOR prototype reduced those explicit dataset chunk bytes from
11220817 to2230844 and passed700 persisted bitwise selection checks. It is not
implemented in the product and is not a scale archive saving claim. See the
living task note for the adopted next slice, exact process times and preserved
reporting failure. The independent Claude Opus5.5/high review has not run:
automatic approval rejected private source transmission pending explicit
source-sharing permission. The goal remains active and incomplete.
