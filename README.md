# ANYpost

ANYpost stores engineering results as atomic, model-bound HDF5 archives and reads selected scalar fields or stable envelopes. Engineering values remain float64. Importing ANYpost requires NumPy/h5py; stored archives do not require ANYsolver or its lightweight runtime.

```python
from anypost import write_results, open_results
write_results("analysis.anypost.h5", capsule)
with open_results("analysis.anypost.h5", expected_model_identity="model-42") as results:
    descriptors = results.components()  # metadata only
    cases = results.cases()             # includes failure diagnostics
    values = results.read_field("case-1", "stress", "von_mises")
    envelope = results.read_envelope("stress", "von_mises")
```

`components(quantity_id=None)` and `cases()` return independent metadata dictionaries. `geometry()` returns shared immutable arrays. `associations(quantity_id, component, selection=None)` preserves flattened sample order, including repeated entity IDs. Selections accept slices or integer indices, including duplicates and reordered indices. No averaging occurs. Reader metadata preserves solver identity, routes, units, basis, sample descriptors and case diagnostics. Explicit stored rows override response reconstruction for their exact case/component.

The default result/basis cache ceiling is 2 GiB. `working_bytes` defaults to 256 MiB and is separately enforced before output allocations; requests too large for the working cap must use a spatial selection. Recovery and envelope evaluation use spatial chunks. The working cap accounts managed numeric buffers and selections; Python/HDF5 implementation overhead and the OS file cache are separate from these diagnostics. `diagnostics` reports resident/peak cache bytes, reads, reconstructions and accounted working bytes. `set_external_cache_bytes(owner_key, byte_count)` registers retained rendering buffers under the same cache ceiling; release with zero. Geometry is outside the result cache.

The base package supports Python 3.10+; the optional solver runtime requires Python 3.11+. Offline reconstruction lazily imports `anysolver_results`, checks its runtime identity, and admits completed certified cases only. ANYpost contains no element recovery equations. Entirely stored data needs no runtime; missing runtime raises `RuntimeDependencyError`. Failed cases remain discoverable and are excluded from envelopes. Exact ties use the first requested admitted case unless descriptor `tie_policy="last"` records a last-requested override. Solver descriptors may record `tie_policy="exact_earliest_input_index"`; then unique nonnegative case `input_index` values choose the earliest original solver input even for a reversed subset. Float64 near ties remain distinct, and failed cases cannot govern. `active_indices` preserves original one-based response-column indices; coefficients are packed in that order and an empty active list returns the baseline once.

ANYfem v1 `anyfem.result` HDF5 sidecars are supported read-only, preserving frame-major fields, component/sample axes, missing-data NaNs, source descriptors, tables, IDs and provenance. Legacy case IDs are zero-based frame index strings. Legacy sidecars carry model/mesh hashes and refer to a separate mesh artifact; `geometry()` is empty. Ambiguous component axes and missing association tables are refused. No pickle is read.

`LatestRequestLoader`, `RetainedResultView` and `PresentedResult` provide parent-owned asynchronous/viewer integration. Rendering conversions use float32; engineering data remains float64. Development tests do not constitute scientific qualification, measured speed claims, Linux headless acceptance or release acceptance.

License: MPL-2.0.

Large exports use the solver's structural sink protocol directly:

```python
from anypost import ArchiveSink
from anysolver import export_recovery_capsule_to_sink
with ArchiveSink("analysis.anypost.h5") as sink:
    export_recovery_capsule_to_sink(
        evaluator, cases, sink=sink,
        model_identity="model-42", mesh_identity="mesh-42", batch_size=32)
```

`ArchiveSink` implements `capsule_begin`, `capsule_case`, `capsule_stored`, and `capsule_end`. It writes geometry/basis immediately, appends coefficients and per-component explicit rows, and retains only compact case/row metadata. Numeric append blocks are at most 256 KiB; component count determines HDF5 groups, not case count. Final response metadata merges with header responses, preserving header-only displacement bases. Final associations and dependencies must match their previously admitted layouts exactly.

`capsule_end` validates the native structure; successful context exit or explicit `close()` atomically publishes it. Exceptions, `abort()`, missing end, duplicate rows, changed layouts or nonempty `stored_superseded_cases` leave the previous destination intact and remove the unpublished temporary file. Superseded rows represent changing solver sample associations and are refused rather than silently discarded. Use `sink.diagnostics` for block-write counts/peak managed write bytes and lifecycle state. The sink module never imports the solver; production drives it through the four callbacks.

Derived-reader computation uses a configurable managed tile target (64 MiB by default), bounded by the remaining `working_bytes`, while HDF5 storage chunks remain 256 KiB. Sibling derived operators use stable tile boundaries for primitive-basis reuse. Exact whole-table dependency association checks run in bounded byte-accounted cached blocks, with compact proofs bound to the immutable reader/model/mesh. Cache eviction releases numeric association blocks; the completed source-bound proof remains valid. Discovery still performs no field reads. Diagnostics add association reads/hits/verifications and computation tile counts.

Native archives are self-contained. External/soft HDF5 links, virtual datasets and raw external-storage datasets are refused before field or geometry access, preventing absent backing files from producing fill-zero engineering success.


When the lightweight runtime supplies the additive `prepare_response` and `reconstruct_prepared_response` pair, ANYpost caches its immutable baseline/selected-column object in the same byte-accounted LRU. Cache identity includes model, mesh, component, spatial grid and original active-column tuple; cases share a prepared block without sharing coefficient values. The runtime continues to own validation and numerical reconstruction. Preparation reserves source/conversion/selected matrices conservatively before allocation, and verifies exact reported numeric buffer bytes. Older runtime builds retain the original reconstruction path; warm performance acceptance requires the measured runtime capability. An incomplete additive API or invalid buffer accounting is a typed refusal.

Actual HDF5 I/O/checksum failures raise typed `ValidationError` with the original `OSError` chained as the cause. A field-local damaged chunk does not hide discoverable case/component metadata; constructor failures close their opened file handles. Native coefficient/association/geometry paths and legacy field/table/association paths share this error boundary.

Native samples remain in explicit flattened association order. A selection such as `np.arange(0, native_row_count, 7)` computes only those requested rows and their dependencies; association equivalence is still verified for the full source tables. Regular positive index strides use a single HDF5 hyperslab per computation tile. Reordered, reverse and duplicate selections retain the sorted-unique/inverse path, preserving requested order and repeated values. No implicit averaging occurs.

Native `anypost.results/1` response matrices may use the optional `uniform_samples/1` storage codec. If the leading association ID forms equal consecutive groups of more than one row, with unique group IDs, the writer stores `(entity_count, samples_per_group, patterns+1)` response datasets in lossless chunks spanning one sample position. This is a storage heuristic for either one-dimensional IDs or the first column of a multi-column association table; all association columns and their source order are preserved and it implies no physical interpretation. Nonuniform, nonconsecutive, unique-ID or empty layouts retain flat response matrices. Explicit stored case rows retain their flat source order and override reconstruction as before.

The reader maps requested logical rows to their group and sample position, reads only those response rows, and restores reverse/arbitrary/duplicate request order before calling the unchanged lightweight runtime. Decoded values remain exact float64 source values; no averaging or source equations change. Codec shape and metadata are admitted at open; the complete association digest, equal-group layout and unique group IDs are proved in bounded temporary buffers before response access, with compact immutable proofs surviving `clear_cache()`. Unknown codecs, malformed shapes/layouts, changed association bindings and damaged checksum chunks are typed refusals. Flat native and legacy archives remain readable. Splitting sample positions into separate chunks can materially increase archive size when the original data compress repeated sample rows well; selective-read speed and archive size should both be measured on representative data.

The optional `results` dependency admits ANYsolver-results major versions 1 and 2 (`>=1.0.0,<3`); that runtime requires Python 3.11 or newer, while the stored-only ANYpost package supports Python 3.10 or newer.

UI hosts can prefetch geometry and descriptor snapshots from the same live reader on their I/O executor, then attach the retained view on the viewer owner thread:

```python
geometry = reader.geometry()          # I/O executor
descriptors = reader.components()
view = RetainedResultView(            # viewer owner thread
    reader, viewer, dispatch,
    geometry=geometry, descriptors=descriptors)
```

The optional snapshots avoid calling `reader.geometry()` and `reader.components()` during owner-thread attachment. The caller continues to own the reader lifetime.

Current Windows/Linux evidence, candidate limitations and the outstanding physical-scale acceptance gate are recorded in [docs/VERIFICATION.md](docs/VERIFICATION.md). The local wheel bundle and SHA256 manifest are at C:/ANYworkspaceWork/anypost-20261012/delivery. `ArchiveSession.prepare` runs on the host I/O executor; pass its single-use result to `ArchiveSession(prepared, viewer)` on the viewer thread. Standalone `anyfem-archive` performs this prefetch before creating its GUI. Benchmark tooling additionally needs `psutil` and `threadpoolctl`; product imports do not require those benchmarking tools.

`write_results(..., stored_encoding="auto")` and `ArchiveSink(..., stored_encoding="auto")` may encode explicit stored rows using `xor_ordered_ieee754/1` when that component has a compatible response basis. `stored_encoding="plain"` retains ordinary float64 stored datasets for comparison or compatibility. Derived-only fields without their own response basis remain plain. Stored case/component overrides retain their source classification and metadata; encoding does not turn them into reconstructed or certified results. A response discovered after a streaming component's plain rows were appended leaves that component plain.

The codec stores uint64 XOR corrections under the ordinary two-dimensional `values` dataset with a whitelisted `codec` attribute. Its storage-only predictor uses the case's original one-based active-column order (absent activity means all columns), separate float64 NumPy multiply then add operations, and the baseline once at the end. An empty active list copies the baseline bits, including signed zero. Local floating-point error handling and canonical quiet-NaN predictor bits (`0x7ff8000000000000`) make extreme finite inputs encodable; the predictor is never exposed as an engineering result. Decoding restores the actual finite source float64 bits exactly, including signed zeros and source rounding, for reordered, reversed and duplicate selections. It requires no solver runtime for a stored row. Unknown codecs or incompatible/missing basis, coefficient or case metadata refuse with `ValidationError`; checksum failures retain the chained HDF5 cause.

Encoding holds one bounded spatial basis tile and visits the incoming case rows, with an 8 MiB managed work ceiling and lossless LZF/shuffle/Fletcher32 chunks up to 256 KiB; `compression=None` remains supported. No unbounded writer basis cache or explicit payload is retained after a sink callback. Selected decoder basis/fields and coefficients share the reader LRU and conservative working reservations; `diagnostics["stored_correction_decodes"]` distinguishes the byte codec from engineering reconstruction. Storage savings and performance need representative matched measurements; a two-element captured export does not establish real-scale acceptance.

`ArchiveSink.capsule_failed(record)` is an optional terminal callback alongside the four existing callbacks. It remains available after abort and atomically preserves one unique same-directory `<archive>.<attempt-id>.failure.json` artifact containing the source finite JSON diagnostics, original input identities/status and compact completed case summaries. `sink.diagnostic_path` exposes that artifact; the previous archive is preserved and incomplete HDF5 is never published. Direct context/close failures also persist available summaries, marking unknown complete input inventories explicitly. If original diagnostics cannot be represented as finite JSON, an incomplete diagnostic artifact and `diagnostic_error` explain the persistence failure; a typed callback error is raised and context cleanup preserves the primary exception. Separate attempts never overwrite earlier evidence.


Retained beam geometry may include `line_to_element`, one dense element-table
index per line. The presentation adapter passes this source mapping through
alongside `triangle_to_element`; fields are reordered by explicit source IDs
before the existing float32 display conversion. Worker-prefetched geometry and
descriptors remain supported. ANY3dView GPU hosts render mapped element scalars
or interpolate provided node endpoint scalars, preserving line primitive keys
and exposing mapped element IDs for fallback picks. Unsupported software line
fields are refused before attachment/update. Reader/cache and float64 engineering
values are unchanged; retained display conversions still reserve reader-cache
bytes and are released on close.


### Shared case metadata and bounded startup

Long top-level `reaction_node_ids` lists of plain integers are stored once per
exact canonical JSON value using `reaction_node_ids/1`. Order, duplicates,
negative IDs and large Python integers are preserved. Nested `solver_info`
values remain original metadata. This optional representation uses native
`anypost.results` version2; ordinary small/pool-free archives stay version1.
Existing version1 archives and ANYfem legacy sidecars retain their meaning.
Older native readers explicitly reject version2. This is a pre-release storage
format evolution; the solver runtime identity remains `anysolver-results/1`.

`case_summaries()` lists compact identities/status/routes and control fields.
`metadata_summary()` includes the fixed source header and bounded case listing;
its explicit `cases_complete` flag is true only when small pool-free detailed
records fit the conservative prefetch bound, and always false for pooled data.
Use these paths for UI startup. `case_metadata(case_id)` returns one independent
original detailed record. Existing `cases()` and `metadata()` remain detailed;
pooled detail expansion exceeding `working_bytes` raises `WorkingSetError`
before allocating all lists. Use summaries or one-case queries instead.

Native2 admission verifies codec, fixed path/null placeholders, references,
model/mesh binding, canonical integer tokens, table counts and SHA256 digests.
Checksummed byte tables are verified incrementally without retaining decoded ID
lists; no archive-supplied code executes. Engineering replay and envelopes use
compact records and never expand these lists. Unique tables are written directly
while the sink retains compact cases/ref/digest metadata, not per-case ID copies.
Failure artifacts retain unchanged primary source diagnostics; completed records
are explicitly labelled summaries when pooling was active and omit the large ID
field. They never reference a deleted temporary archive.
### Native sample stored corrections

Automatic stored encoding uses `xor_ordered_ieee754_uniform_samples/1` when the
response basis has verified uniform source associations. Its uint64 correction
masks have physical `(case, entity, sample)` shape and lossless chunks containing
one case and one sample. Logical fields retain their original flattened source
order. Positive station strides read only that station; arbitrary, reversed and
duplicate selections restore exactly the requested order. Existing flat
`xor_ordered_ieee754/1` and plain float64 archives remain readable. `plain` still
opts out of correction encoding; its chunks now contain one case.

The ordered storage predictor and decoded source float64 bits are unchanged.
Encoding retains an 8MiB working bound. Derived computation targets at most
64MiB per tile, further limited by the configured working cap and conservative
full dependency/scratch reservations. Predictor bases, including their Fortran
packing, remain under the managed cache ceiling. These storage choices do not
certify source arithmetic or introduce physical recovery formulas.

### Source runtime pairing and case admission

Capsule1 uses `anysolver-results/1` and the legacy BLAS combination. Capsule2
requires `anysolver-results/2` and `ordered_float64/1`. Native metadata retains
`capsule_version` and `combination`; the independent native archive versions1/2
still distinguish pooled metadata. Old native metadata without these keys keeps
its runtime1 meaning. New runtime2 can advertise and replay both source contracts;
an older runtime1 remains sufficient for legacy archives. Explicit stored rows
need no solver runtime. The optional results dependency permits major1 or2.

`case_components(case_id, quantity_id=None)` reports each case's stored,
reconstructable or unavailable path without reading fields or importing physics.
Source `unavailable_components` always refuses a field. A missing explicit row
cannot fall back to a basis when `portable=False`; displacements additionally
honour an explicit false/null `displacement_portable`. Unknown identities raise
`UnknownCaseError`, also a `ValidationError` and `KeyError`. Detailed failed case
metadata remains available for diagnostics; field/discovery requests refuse it.

Envelopes report `requested_case_ids`, `admitted_case_ids` and `failed_case_ids`
(`excluded_case_ids` alias). Default envelopes exclude failed cases visibly;
`strict_failed=True` refuses them. All-failed requests carry those IDs on the
error. Missing/unavailable fields never silently disappear. Exact input-index
and float64 near-tie rules remain unchanged.

Field, association, geometry and envelope reads accept `cancel_event` with
`is_set()`. Cancellation raises `concurrent.futures.CancelledError` between
bounded tiles/proofs and before cache publication or returning a result. A
currently running HDF5/NumPy call finishes its bounded work first. Completed
immutable blocks may remain cached; partial fields are never published.
Presentation forwards its loader event only to readers explicitly advertising
`supports_cancellation=True`, preserving custom providers without that keyword.
Capsule2 geometry maps require integer dense indices, matching topology lengths
and explicit element IDs; legacy geometry validation keeps its prior meaning.

### Required v2 controls, cache identity and recovery context

Every capsule2 case requires explicit `portable` and `displacement_portable`
flags, including stored and failed cases. Missing flags refuse writer, streaming
and reader admission. Replay requires positive portability; the canonical
`displacement:` source key remains authoritative even when its quantity label
is renamed. Capsule1 missing-flag behavior and arithmetic stay unchanged.

Basis and prepared caches use disjoint tagged range/explicit-index identities,
so a slice such as `0:48:3` cannot alias the three IDs `[0,48,3]`. Runtime
reconstruction/derivation refusals raise `RecoveryError` with case, component,
runtime, combination and the original exception cause. Programmer, cancellation,
identity and working-cap errors keep their original types.

Diagnostics expose `capsule_version`, `runtime`, `combination` and
`response_layout_deterministic`, scoped to the response combination. Legacy BLAS
remains layout dependent; this label does not certify arbitrary derived physics
or numerical backends. Envelopes still refuse completed unavailable fields and
now identify all such inputs through the readonly `unavailable_case_ids` tuple
in source input order. No completed unavailable case is silently excluded.

Retained presentation now retires/cancels its worker before releasing reader cache reservations. Failed attachments remove only their own mesh; the caller retains reader ownership. Domain switches use the additive ANY3dView atomic scalar transaction; older handles explicitly refuse cross-domain changes. `open_results(..., derivation_workers=None)` selects two principal-derivation workers only for runtime capability `DERIVATION_EXECUTION_API=1`; explicit1/2 are accepted, older runtimes remain serial. Diagnostics report requested/effective workers and parallel calls. Capsule2 nodal reconstruction requires canonical source displacement/reaction keys; stored arbitrary nodal outputs and legacy1 remain supported.

Capsule2 reconstructable fields use closed source wire classes: canonical node displacement/reaction keys, or element stress fields, or element quantity=component=key fields. Noncanonical locations and aliases refuse at write/open. Future nonstress element aliases require a versioned source extension; stored arbitrary fields and capsule1 meanings are preserved.

Uppercase nodal DOF component labels UX/UY/UZ/RX/RY/RZ/FX/FY/FZ/MX/MY/MZ are reserved for canonical node replay in capsule2, even when quantity/location are disguised as stress/element. Future element outputs using these labels need a versioned source-class extension.

ANYpost 0.1.2 includes `open_memory_results(provider, cache_bytes=2<<30,
working_bytes=128<<20, expected_model_identity=None)`. The provider owns recovery
and lifetime; closing the reader releases its buffers and never closes the provider.
No query creates a temporary archive or imports ANYfem or a solver runtime.

`provider.metadata()` returns model identity, cases and descriptors without loading
fields. Case identity is `case_id` (or `id`), status is preserved, and completed
legacy cases need no invented certification. Descriptors declare quantity_id,
component or components, location, unit/basis/provenance, persistent integer ids
(or spatial_count), dtype (float64 default), and trailing value_shape/sample_shape.
Ragged padded native rows require explicit sample_counts; only the counted prefix
in C order is actual data. A descriptor/provider must declare provider_work_bytes,
including recovery workspace (explicit zero for already-retained extraction).
Dynamic `field_descriptor(case_id, quantity_id, component)` or case_descriptors
may supply per-case counts/shapes; the static provider bound is admitted before
calling the dynamic descriptor. Providers must honour declarations and apply
their own recovery resource policies; the reader's cap is not a process-RSS claim.

`read_field(case_id, quantity_id, component, selection=None, *, reduction=None,
nan_policy='propagate', empty_policy='nan', cancel_event=None)` forwards only the
selected spatial rows to provider.read_field with the same selection/cancel_event
keywords. Selections preserve reverse/duplicate order. Returned dtype and shape
must match the declared layout; no display conversion or engineering interpolation
occurs. Independent bytes snapshots and fresh public array headers isolate the
LRU from caller/source mutation. Metadata arrays count as non-evictable cache
reservations; clear_cache clears fields and retains those/live external buffers.
Provider data is a stable snapshot for the reader lifetime; clear_cache or open a
new reader after changing the underlying provider data.

Explicit native reductions are mean/min/max/max_abs; signed max_abs selects the
first actual sample at an exact absolute tie. NaN policy is propagate (default),
omit or raise; padded entries outside sample_counts do not participate. Empty
actual sample rows default to NaN; caller-requested empty_policy='zero' reproduces
historical reduction behavior only for present empty rows, never absent fields.
`magnitude` requires declared vector_components=('ux','uy','uz') and shape(N,3)
with all three components, using np.linalg.norm(axis=1); stress triplets are refused.

`envelope` returns the existing min/max Envelope with governing IDs and requested,
admitted and failed case coverage. `read_envelope(..., mode='min'|'max'|'max_abs')`
returns EnvelopeSelection.values/case_ids and the same coverage. Source input_index
controls exact_earliest_input_index ties; first/last follow caller order. Failed
cases are excluded/reported, strict_failed refuses them, and missing successful
fields refuse with unavailable_case_ids. Streaming holds one requested case field,
not all case fields. With nan_policy=propagate, any present source NaN propagates
regardless of case order; its governing ID is empty. Omit excludes actual NaNs;
raise refuses them. Optional geometry,
associations, read_history and query methods return bounded immutable snapshots;
provider-owned recovery for these methods must already enforce its resource bound.

Per-case descriptor `present` is an explicit boolean spatial mask, default alltrue.
Absent entities always remain missing even with caller-requested emptyzero.
Envelopes skip absent entities; any *present* sourceNaN propagates without
a finite governing ID. Presentempty samples remain distinct from absent fields.
Public legacy `read_history(key, selection=None, cancel_event=None)` selects the
frame axis of existing numeric histories/<key> datasets or original x/y history
groups (returned as a readonly (x,y) pair) with working admission,
exact reverse/duplicate order, typed missingdata and cooperative cancellation.

MemoryResultsReader.read_stored_field accepts the same selection/reduction/NaN/
empty/cancellation arguments as read_field. It exposes only explicitly declared
availability='stored' snapshots, including incomplete source cases, without
changing status, commitment, certification or envelope eligibility. Reconstructable,
unavailable and unknown routes refuse before provider recovery. Ordinaryread_field
still refuses incomplete cases and infinity. Explicit stored-only access preserves
actual source IEEE infinity for display compatibility, without certification or
envelope admission. This accessor is not compact-replay eligibility.

Legacy sidecar quantities preserve their own declared frame axes and local row order. When an exact unique association to the root frame table cannot be proved, descriptor `frame_axis="quantity"` and `case_ids` name quantity-scoped stored rows. These IDs do not add root cases or frames and cannot be used for other quantities. `case_metadata()` preserves the original source frame and partial status; normal reads refuse partial results, while explicit `read_stored_field()` reads actual stored snapshots.

Undeclared one-dimensional legacy fields are independent spatial rows; scoped metadata leaves the source frame undeclared rather than associating it to a root frame. Original descriptor transport source_key names the HDF group and preserves optional original key. History discovery reads only headers and exposes the common x/y shape. Pending cancelled loader tasks retire without a false wait timeout; active tasks must still retire within the close deadline.

Unreduced provider envelopes combine entity presence with each case's declared sample_counts in the native C-flattened trailing-slot order. Missing padding is excluded from comparisons and NaN policy; actual source NaNs still propagate with no finite governing ID. Scratch for selected counts, ordinal indices and the sample mask is included in working admission before provider field reads. Reduced queries keep their existing actual-sample semantics.

Provider index-selection key payloads count toward cache residency together with field values, including eviction, resize and oversized-value decisions. Partial-presence reductions include the live original source array as well as the selected copy and output. Descriptor keys must be nonempty and unique across expanded components; shared explicit keys are refused rather than silently renamed.

Optional provider geometry/associations/history/table/query transports require a nonnegative integer provider_work_bytes declaration. It bounds every new provider allocation through return; zero explicitly means pre-retained results. ANYpost reserves this workspace before invoking the provider. Optional passthrough_work_bytes(method_name,*args,**kwargs) supplies a request-specific bound; its own work is admitted under the global bound first. After return, a conservative expanded graph bound (including repeated aliases) admits source, snapshot and public-return coexistence before copying. Cyclic graphs are refused. Snapshot arrays remain immutable byte-backed views and maps immutable. Physics/model/pre-retained source residency remains provider-owned and separately accounted; this contract does not bound arbitrary dishonest callbacks or total process RSS.

Provider boundary details (0.1.2): `empty_policy="raise"` raises public `EmptySampleError` (an `UnavailableFieldError` subtype); envelopes propagate this explicit reduction refusal immediately. All-failed and strict-failed `FailedCaseError` expose `failed_case_ids` and `requested_case_ids`. A case sequence must not be a string; an empty index list selects zero rows.

Descriptor ID/count/presence vectors are normalized and validated once, with complete-vector conversion and validation workspace admitted before allocation. Metadata source/snapshot coexistence is conservatively admitted over the unique object graph; shared immutable payloads remain shared, and cycles are refused before copying. Metadata cache reservations use a private owner namespace that public external reservations cannot overwrite.

Static `availability="stored"` is the provider's assertion that actual source snapshots exist for each covered case unless a case override says unavailable. It does not assert completion, solver commitment or certification. Explicit stored access preserves source statuses and provenance. Core reader operations are synchronous and serialized; applications own worker coordination and must retire queries before closing or publishing through their own nonblocking UI gate.

A live reader component has fixed persistent entity IDs, spatial count, floating dtype, units, basis, location and component definition. Static/dynamic case descriptors may change presence and native sample counts/shapes, but incompatible axis or precision changes are refused before field recovery. Unknown case-descriptor cases/keys refuse at open. Declared unavailable fields are refused before recovery and reported in envelope coverage.

Query policies are validated before recovery. For raw unreduced reads, `nan_policy="raise"` inspects actual native slots only; other NaN policies and `empty_policy` do not transform raw native samples. Empty policy applies when an explicit reduction is requested. Normal engineering infinity admission likewise inspects only present actual slots: absent rows and padding are not engineering evidence. Raw padding bytes remain raw; explicit stored-only reads retain their existing IEEE snapshot policy.

Selections are admitted before list conversion, integer normalization, validation temporaries and cache-key copying. Immutable metadata size is cached, with constructor normalization tracked incrementally; dynamic descriptor reads do not traverse the entire case table. Empty envelope case sequences are request-shape errors; single failed-case errors preserve exact failed/requested IDs.

Case `unavailable_components` must be a unique sequence of known descriptor keys. Explicit spatial counts must be integers consistent with IDs; shape aliases must agree. Unrelated provenance/extension keys remain permitted. Sampling definition is fixed for a reader component. A missing dynamic descriptor (`KeyError`) is unavailable case-field evidence, matching missing field queries.

For reduced envelopes, a row with zero actual samples under `empty_policy="nan"` contributes neither a governing value nor source-NaN evidence. Later finite samples may govern regardless of case order. Explicit `zero` retains its requested zero/tie behavior; `raise` remains `EmptySampleError`. Actual source NaNs still follow the selected NaN policy. Raw unreduced/native padding semantics remain unchanged.

Descriptor IDs must be an integer vector even when an explicit spatial count is supplied. When both shape aliases are present they must agree; a per-case override cannot silently replace one alias while retaining a conflicting other alias.

`empty_policy="zero"|"raise"` applies to structurally present rows with zero actual samples. Under `nan_policy="omit"`, counted rows whose actual samples are all NaN produce NaN, without invented zero or empty-row refusal; omit envelopes then permit another finite case to govern. Default propagate and raw native semantics are unchanged.

Legacy ANYfem sidecar metadata copies retain raw NaN frame/ordinal labels and independent array/container copies. Identical full root/quantity axes pair NaNs by original row order; independent or ambiguous subset axes retain quantity-scoped identities. Public discovery needs no field-value loads. This compatibility rule does not permit nonfinite JSON in native ANYpost archives.

Association-pool storage may hard-link equal tables. Derived admission compares complete persisted dependency mappings before any field reconstruction, independently of requested selection. A corruption test for one dependency must first detach that link; mutating a shared table changes every linked mapping and does not create a differing dependency.
