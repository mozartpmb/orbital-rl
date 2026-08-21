"""ctypes binding for the J2 propagation kernel (`nav_j2_kernel.c`).

WHY ctypes AND NOT A setup.py EXTENSION. The kernel is called ONCE PER PREDICT
with the whole batch, so per-call FFI overhead is amortised over n*13
propagations and is unmeasurable. In exchange the build stays a single gcc line
with no numpy C-API version coupling and no edit to the extension list that the
bit-exact env anchors run through. If this ever moves into the hot inner loop
(it should not — that would mean the batch shrank to ~1), revisit.

The module NEVER raises on a missing .so at import: `available()` reports, and
the caller falls back to Python. A missing shared object must degrade to the
oracle implementation, not take a training run down.
"""
import ctypes
import os

import numpy as np

_SO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'nav_j2_kernel.so')
_lib = None
_err = None

try:
    _lib = ctypes.CDLL(_SO)
    _lib.stm_fd_j2_batch.restype = None
    _lib.stm_fd_j2_batch.argtypes = [
        np.ctypeslib.ndpointer(np.float64, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_int, ctypes.c_double,
        np.ctypeslib.ndpointer(np.float64, ndim=3, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(np.uint8, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(np.float64, ndim=2, flags='C_CONTIGUOUS')]
    _lib.propagate_cartesian_j2_batch.restype = None
    _lib.propagate_cartesian_j2_batch.argtypes = [
        np.ctypeslib.ndpointer(np.float64, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_int, ctypes.c_double,
        np.ctypeslib.ndpointer(np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(np.uint8, ndim=1, flags='C_CONTIGUOUS')]
except OSError as e:                                    # pragma: no cover
    _err = str(e)


def available():
    return _lib is not None


def why_unavailable():
    return _err or 'nav_j2_kernel.so not built'


def _c_in(X):
    """float64 C-contiguous (n,6). `ascontiguousarray` is not decoration: the
    kernel indexes X + 6*i, so a transposed or strided view would be read as
    garbage WITHOUT erroring — the classic silent batch-layout bug. The
    permutation-invariance gate exists to catch this class."""
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim != 2 or X.shape[1] != 6:
        raise ValueError(f'expected (n,6), got {X.shape}')
    return X


def stm_fd_j2_c(X, dt):
    X = _c_in(X)
    n = X.shape[0]
    Phi = np.empty((n, 6, 6), dtype=np.float64)
    ok = np.empty(n, dtype=np.uint8)
    Y = np.empty((n, 6), dtype=np.float64)
    if n:
        _lib.stm_fd_j2_batch(X, n, float(dt), Phi, ok, Y)
    return Phi, ok.astype(bool), Y


def propagate_cartesian_j2_c(X, dt):
    X = _c_in(X)
    n = X.shape[0]
    Y = np.empty((n, 6), dtype=np.float64)
    ok = np.empty(n, dtype=np.uint8)
    if n:
        _lib.propagate_cartesian_j2_batch(X, n, float(dt), Y, ok)
    return Y, ok.astype(bool)
