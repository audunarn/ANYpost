# Principal display axes

`anypost.principal_axes(tensors, basis=None, cancel_event=None,
working_bytes=67108864)` delegates to the NumPy-only ANYsolver-results runtime.
Inputs must be admitted finite symmetric stress tensors with known sample and
coordinate associations. The facade does not recover missing stress components.

The returned immutable arrays are `values[..., rank]` in descending order,
`axes[..., rank, xyz]`, and `unique[..., rank]`. Values use the existing canonical
principal-stress kernel. Axes represent unoriented directions; arrow signs are
not physical flow directions. Numerically unresolved repeated-eigenvalue axes
are marked false and filled with NaN. The display diagnostic uses 64 float64 eps
times the maximum eigenvalue magnitude; it does not change scientific acceptance
or any existing stress tolerance.

Optional proper-orthonormal basis columns represent local axes in global
coordinates. Cancellation is checked before admission and between bounded
independent tensor blocks. The working estimate is admitted before conversion
of ndarray/nested-sequence tensor inputs. Outputs publish only after completion.
Malformed inputs, unavailable runtime capability, cancellation and working-limit
refusal use the existing ANYpost typed errors. No archive-derived executable
operation is registered and importing ANYpost does not import solver physics.

Verification consists of rotated analytic eigen-residuals, canonical value
agreement, repeated/zero tensors, basis admission, source immutability, typed
refusal, pre-conversion working admission and cancellation. This feature does
not broaden solver qualification.

Owner integration preserves that display convention and canonical value kernel.
Shape counts use exact Python integers, malformed dimensions refuse, and tensor
and basis metadata must be compatible before conversion. The working estimate
is an admission policy, not a RAM reservation or measured peak-memory guarantee.
Actual ndarray shape comes from the base NumPy descriptor; shape overrides are
ignored without execution. This adds no arbitrary-conversion or concurrent-input
mutation purity guarantee.
