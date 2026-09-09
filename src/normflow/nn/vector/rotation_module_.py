# Copyright (c) 2026 Javad Komijani

"""This module contains new neural networks for transforming vectors.

The classes defined here are subclasses of `Module_`, and like it, the
trailing underscore implies that the associated forward and reverse methods
handle the Jacobians of the transformation.
"""

# pylint: disable=invalid-name, relative-beyond-top-level

from typing import Tuple

import torch

from .._core import Module_


__all__ = ["VectorRotation_"]


# =============================================================================
class VectorRotation_(Module_):
    """A learnable rotation acting on a vector in R^n, along its last axis.

    The rotation matrix R is parametrized as `R = exp(A - A^T)`, where A is
    a free n x n weight matrix; R is exactly orthogonal (R^T R = I, det R = 1)
    for any value of the parameters, so the transformation is exactly
    invertible via R^T. In particular, since |Rx| = |x|, it maps any sphere
    centered at the origin (e.g. the unit sphere |x| = 1) to itself.

    Because R is orthogonal, |det R| = 1, so it preserves the Lebesgue
    measure on R^n exactly and the log-Jacobian contribution of this
    transformation is exactly zero.

    Parameters
    ----------
    n : int
        Dimension of the ambient space (length of the input's last axis).
    """

    def __init__(self, n: int):
        super().__init__()
        self.n = n
        self.weight = torch.nn.Parameter(1e-2 * torch.randn(n, n))

    def rotation_matrix(self) -> torch.Tensor:
        """Return the rotation matrix `R = expm(A - A^T)`."""
        a = self.weight - self.weight.T  # skew-symmetric generator
        return torch.matrix_exp(a)  # exact element of SO(n)

    def forward(
        self, x: torch.Tensor, log0: torch.Tensor | float = 0
    ) -> Tuple[torch.Tensor, torch.Tensor | float]:
        """Apply the rotation `y = R x`.

        Args:
            x: Input vector(s).
            log0: Log-Jacobian of past transformations. (Default is 0.)

        Returns:
            The rotated vector `y` and the (unchanged) log-Jacobian.
        """
        y = x @ self.rotation_matrix().T
        return y, log0  # rotations preserve the Lebesgue measure: logj = 0

    def reverse(
        self, y: torch.Tensor, log0: torch.Tensor | float = 0
    ) -> Tuple[torch.Tensor, torch.Tensor | float]:
        """Apply the inverse rotation `x = R^T y`.

        Args:
            y: Input vector(s).
            log0: Log-Jacobian of past transformations. (Default is 0.)

        Returns:
            The rotated vector `x` and the (unchanged) log-Jacobian.
        """
        x = y @ self.rotation_matrix()
        return x, log0  # rotations preserve the Lebesgue measure: logj = 0
