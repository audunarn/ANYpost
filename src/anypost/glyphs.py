"""Typed admission facade for the canonical result-runtime principal axes."""
from .errors import RuntimeDependencyError, ValidationError, WorkingSetError, RequestCancelled


def principal_axes(tensors, *, basis=None, cancel_event=None, working_bytes=67108864):
    """Return descending principal values, rank-major axes and uniqueness flags.

    Tensor components and basis must already come from an admitted canonical
    field; this function does not recover stresses or infer sample associations.
    """
    try:
        from anysolver_results.principals import principal_axes as derive
    except ImportError as error:
        raise RuntimeDependencyError('principal axes require a runtime with principal_axes capability') from error
    try:return derive(tensors,basis=basis,cancel_event=cancel_event,working_bytes=working_bytes)
    except InterruptedError as error:raise RequestCancelled(str(error)) from error
    except MemoryError as error:raise WorkingSetError(str(error)) from error
    except (ValueError,TypeError,OverflowError) as error:raise ValidationError(str(error)) from error
