# Copyright (c) 2026 Javad Komijani

"""This module has utilities to deal with euler decompostions of matrices."""

import torch

from lattice_ml.linalg import sun_to_euler_angles, euler_angles_to_sun


__all__ = ["SUnMatrixEulerParametrizer"]


# =============================================================================
class SUnMatrixEulerParametrizer:
    r"""For Euler decomposition of SU(2) and SU(3) matrices.

    The decomposition itself, and its log-Jacobian, live in `lattice_ml.linalg`
    next to the coordinates they belong to; this class is the adapter onto the
    `matrix2param_` / `param2matrix_` protocol that `MatrixModule_` expects.

    The default `coords='uniform'` gives coordinates that are exactly uniform
    on [0, 1]; so the log-Jacobian is identically zero.
    """

    def __init__(self, coords='uniform'):
        self.coords = coords

    def matrix2param_(self, matrix):
        """Return the Euler coordinates of `matrix` and the log-Jacobian."""
        # channel_axis=-1 explicitly to stack coordinates on a channel axis.
        param, logj = sun_to_euler_angles(
            matrix, coords=self.coords, channel_axis=-1, return_logj=True
        )
        return param, sum_density(logj)

    def param2matrix_(self, param, reduce_=False):
        """Reverse of `matrix2param_`."""
        matrix, logj = euler_angles_to_sun(
            param, coords=self.coords, channel_axis=-1, return_logj=True
        )
        if reduce_:
            raise ValueError("reduce_=True is not supported")
        return matrix, sum_density(logj)


# =============================================================================
def sum_density(x: torch.Tensor):
    """Compute the sum over all, but the batch, axes."""
    ndim = x.dim()
    return x if ndim < 2 else torch.sum(x, dim=list(range(1, ndim)))
