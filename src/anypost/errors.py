"""Typed errors at archive, qualification, and resource boundaries."""
class ResultsError(Exception):
    """Base error for ANYpost."""
class ValidationError(ResultsError, ValueError):
    """Malformed archive or capsule."""
class IdentityError(ValidationError):
    """Expected model or mesh does not match the archive."""
class UnavailableFieldError(ResultsError):
    """The requested quantity has no admitted recovery path."""
    def __init__(self, *args, unavailable_case_ids=()):
        super().__init__(*args)
        self._unavailable_case_ids = tuple(unavailable_case_ids)
    @property
    def unavailable_case_ids(self): return self._unavailable_case_ids
class EmptySampleError(UnavailableFieldError):
    """An available native sample row is empty under explicit raise policy."""
class FailedCaseError(ResultsError):
    """A case did not complete; diagnostics remain in its metadata."""
    def __init__(self, *args, failed_case_ids=(), requested_case_ids=()):
        super().__init__(*args)
        self.failed_case_ids = tuple(failed_case_ids)
        self.requested_case_ids = tuple(requested_case_ids)
class RuntimeDependencyError(ResultsError, ImportError):
    """The admitted lightweight runtime is absent or incompatible."""
class WorkingSetError(ResultsError, MemoryError):
    """Request exceeds the explicit transient working byte cap."""
class ClosedReaderError(ResultsError):
    """Reader handle has already been closed."""

class UnknownCaseError(ValidationError, KeyError):
    """Requested case identity is absent from the archive."""

class RecoveryError(ResultsError):
    """The admitted runtime refused a recovery call; its cause is preserved."""
    def __init__(self, message, *, case_id, component_key, runtime, combination):
        super().__init__(message)
        self.case_id = case_id
        self.component_key = component_key
        self.runtime = runtime
        self.combination = combination


from concurrent.futures import CancelledError

class RequestCancelled(ResultsError, CancelledError):
    """A caller cancelled an archive write through cancel_event."""
