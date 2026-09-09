# Copyright (c) 2026 Javad Komijani

"""This module contains context-conditioned neural networks that rotate
vectors via a rotation matrix built from context, rather than via free
trainable parameters (contrast `rotation_module_.py`).

The classes defined here are subclasses of `Module_`, and like it, the
trailing underscore implies that the associated forward and reverse
methods handle the Jacobians of the transformation.
"""

# pylint: disable=invalid-name, relative-beyond-top-level

from typing import Callable, Tuple

import torch

from .._core import Module_


__all__ = ["VectorRotationContextModule_"]


# =============================================================================
class VectorRotationContextModule_(Module_):
    """Conditional rotation of a vector, generalizing `VectorRotation_` to a
    context-dependent rotation matrix.

    Given `R(context) = expm(A - A^T)`, with `A` an n x n matrix built from
    `feature_map_fn`'s output, transforms the input vector by:

        forward: y = R x   (along the last axis)
        reverse: x = R^T y

    `R` depends only on `context`, never on `x`/`y`. Since `R` is orthogonal
    for any value of `A`, this is exactly Lebesgue-measure-preserving (zero
    log-Jacobian); `forward`/`reverse` just pass `log0` through unchanged.

    Parameters
    ----------
    n : int
        Dimension of the rotated vector space.

    feature_map_fn : Callable
        Maps the context to a real feature tensor whose last axis has
        `n * n` entries, reshaped into the free generator matrix `A`.
    """

    def __init__(self, n: int, feature_map_fn: Callable):
        super().__init__()
        self.n = n
        self.feature_map_fn = feature_map_fn

    def _rotation(self, args) -> torch.Tensor:
        """Build `R(context) = expm(A - A^T)` from `args`."""
        if not isinstance(args, tuple):
            args = () if args is None else (args,)
        features = self.feature_map_fn(*args)
        a = features.reshape(*features.shape[:-1], self.n, self.n)
        return torch.matrix_exp(a - a.transpose(-2, -1))

    def forward(
        self, x: torch.Tensor, log0: torch.Tensor | float = 0, args=None
    ) -> Tuple[torch.Tensor, torch.Tensor | float]:
        """Apply the context-conditioned rotation `y = R x`.

        Args:
            x: Input vector(s).
            log0: Log-Jacobian of past transformations. (Default is 0.)
            args: Context passed to `feature_map_fn`.

        Returns:
            The rotated vector `y` and `log0` unchanged.
        """
        r = self._rotation(args)
        y = torch.einsum('...ij,...j->...i', r, x)
        return y, log0

    def reverse(
        self, y: torch.Tensor, log0: torch.Tensor | float = 0, args=None
    ) -> Tuple[torch.Tensor, torch.Tensor | float]:
        """Apply the context-conditioned inverse rotation `x = R^T y`.

        Args:
            y: Input vector(s).
            log0: Log-Jacobian of past transformations. (Default is 0.)
            args: Context passed to `feature_map_fn`.

        Returns:
            The inverse-rotated vector `x` and `log0` unchanged.
        """
        r = self._rotation(args)
        x = torch.einsum('...ji,...j->...i', r, y)
        return x, log0
