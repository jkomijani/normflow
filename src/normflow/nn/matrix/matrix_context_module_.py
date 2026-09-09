# Copyright (c) 2026 Javad Komijani

"""This module contains context-conditioned neural networks that transform
matrices via the group itself, rather than via a parametrize/transform/
reconstruct cycle (contrast `matrix_module_.py`).

The classes defined here are subclasses of `Module_`, and like it, the
trailing underscore implies that the associated forward and reverse
methods handle the Jacobians of the transformation.
"""

# pylint: disable=invalid-name, relative-beyond-top-level

from abc import ABC, abstractmethod
from typing import Callable, Tuple

import torch

from .._core import Module_


__all__ = [
    "GroupAffineContextModule_",
    "SUnAffineContextModule_",
    "UnAffineContextModule_",
]


# =============================================================================
class GroupAffineContextModule_(Module_, ABC):
    r"""Abstract base: conditional group-element multiplication on unitary
    matrices, generalizing an affine "shift" to a compact Lie group.

    Given `R(context) = expm(1j * H(context))`, `H` Hermitian, transforms
    the input matrix `M` by one of three multiplications, chosen at
    construction via `mode`:

        mode='left':      M -> R @ M
        mode='right':     M -> M @ R^dagger
        mode='conjugate': M -> R @ M @ R^dagger

    `R` depends only on `context`, never on `M`. All three modes are
    exactly Haar-measure-preserving (zero log-Jacobian) on any compact
    group: Haar measure is invariant under left translation, right
    translation, and -- as their composition, `g -> h g h^{-1}` -- under
    conjugation too. `forward`/`reverse` just pass `log0` through
    unchanged; there is nothing to derive or compute.

    This is an abstract class: it is generic over the choice of unitary group.
    A concrete subclass must define `assemble_hermitian`, turning the raw
    features from `feature_map_fn` into a Hermitian matrix obeying that group's
    algebra.

    Parameters
    ----------
    feature_map_fn : Callable
        Maps context to a real feature tensor with however many degrees
        of freedom `assemble_hermitian` needs -- see each subclass's
        docstring for the exact count.

    mode : {'left', 'right', 'conjugate'}
        Which multiplication to apply. Default is 'conjugate'.
    """

    def __init__(self, feature_map_fn: Callable, mode: str = 'conjugate'):
        super().__init__()
        if mode not in ('left', 'right', 'conjugate'):
            raise ValueError(
                f"mode must be 'left', 'right', or 'conjugate', got {mode!r}"
            )
        self.feature_map_fn = feature_map_fn
        self.mode = mode

    @abstractmethod
    def assemble_hermitian(self, features: torch.Tensor) -> torch.Tensor:
        """Turn `feature_map_fn`'s raw real output into a Hermitian matrix
        obeying this group's algebra (e.g. traceless, for SU(N))."""

    def _rotation(self, args) -> torch.Tensor:
        """Build `R(context) = expm(1j * H(context))` from `args`."""
        if not isinstance(args, tuple):
            args = () if args is None else (args,)
        features = self.feature_map_fn(*args)
        H = self.assemble_hermitian(features)
        return torch.matrix_exp(1j * H)

    def forward(
        self, M: torch.Tensor, log0: torch.Tensor | float = 0, args=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the forward transformation.

        Args:
            M: Input matrix.
            log0: Log-Jacobian of past transformations. (Default is 0.)
            args: Context passed to `feature_map_fn`.

        Returns:
            The transformed matrix (per `mode`) and `log0` unchanged.
        """
        R = self._rotation(args)
        if self.mode == 'left':
            return R @ M, log0
        if self.mode == 'right':
            return M @ R.adjoint(), log0
        return R @ M @ R.adjoint(), log0

    def reverse(
        self, M: torch.Tensor, log0: torch.Tensor | float = 0, args=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the reverse (inverse) transformation.

        Args:
            M: Input matrix.
            log0: Log-Jacobian of past transformations. (Default is 0.)
            args: Context passed to `feature_map_fn`.

        Returns:
            The inverse-transformed matrix (per `mode`) and `log0`
            unchanged.
        """
        R = self._rotation(args)
        if self.mode == 'left':
            return R.adjoint() @ M, log0
        if self.mode == 'right':
            return M @ R, log0
        return R.adjoint() @ M @ R, log0


# =============================================================================
class SUnAffineContextModule_(GroupAffineContextModule_):
    """
    Concretizes `GroupAffineContextModule_` for SU(N).
    """
    def assemble_hermitian(self, features: torch.Tensor) -> torch.Tensor:
        N = _n_from_feature_dim(features.shape[-1], traceless=True)
        return _features_to_hermitian(features, N, traceless=True)


# =============================================================================
class UnAffineContextModule_(GroupAffineContextModule_):
    """
    Concretizes `GroupAffineContextModule_` for U(N).
    """
    def assemble_hermitian(self, features: torch.Tensor) -> torch.Tensor:
        N = _n_from_feature_dim(features.shape[-1], traceless=False)
        return _features_to_hermitian(features, N, traceless=False)


# =============================================================================
def _n_from_feature_dim(n_feat: int, traceless: bool) -> int:
    """Recover `N` from the feature count of a (traceless) Hermitian
    `N x N` matrix: `N**2` (or `N**2 - 1`, if `traceless`)."""
    N = round((n_feat + int(traceless)) ** 0.5)
    expected = N * N - int(traceless)
    if n_feat != expected:
        raise ValueError(
            f"features has {n_feat} entries, which is not "
            f"N**2{' - 1' if traceless else ''} for any integer N"
        )
    return N


def _features_to_hermitian(
    features: torch.Tensor, N: int, traceless: bool
) -> torch.Tensor:
    """Assemble a batch of real feature vectors into Hermitian `N x N`
    matrices: the first `N` (or `N - 1`, if `traceless`) entries become
    the diagonal, the rest become the real/imaginary parts of the strictly
    upper triangle (mirrored onto the lower triangle by conjugation)."""
    *batch, _ = features.shape
    n_diag = N - 1 if traceless else N
    n_off = N * (N - 1) // 2

    complex_dtype = (
        torch.complex128 if features.dtype == torch.float64
        else torch.complex64
    )

    diag = features[..., :n_diag]
    if traceless:
        diag = torch.cat([diag, -diag.sum(dim=-1, keepdim=True)], dim=-1)

    off = features[..., n_diag:].reshape(*batch, n_off, 2)
    idx = torch.triu_indices(N, N, offset=1)

    H = torch.diag_embed(diag).to(complex_dtype)
    H[..., idx[0], idx[1]] = torch.complex(off[..., 0], off[..., 1])
    H[..., idx[1], idx[0]] = torch.complex(off[..., 0], -off[..., 1])
    return H
