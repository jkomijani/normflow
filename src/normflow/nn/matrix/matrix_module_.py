# Copyright (c) 2021-2026 Javad Komijani

"""This module contains new neural networks for transforming matrices.

The classes defined here are subclasses of `Module_`, and like it, the trailing
underscore implies that the associated forward and reverse methods handle the
Jacobians of the transformation.
"""

# pylint: disable=invalid-name, relative-beyond-top-level, too-many-arguments

from typing import Tuple

import torch

from .._core import Module_


__all__ = ["MatrixModule_"]


# =============================================================================
class MatrixModule_(Module_):
    """A module for transforming matrices.

    Implements only the skeleton: parametrize the input matrix, transform
    the parameters, reconstruct the output matrix. The parametrization
    scheme -- eigendecomposition, Euler angles, or anything else, including
    schemes with no decomposition at all -- is entirely up to `matrix_handle`.
    It also works for U(1) theory: pass `matrix_handle=U1Parametrizer()` from
    `normflow.lib.matrix_handles`, in which case `x` is a plain complex value
    rather than a matrix.

    Parameters
    ----------
    param_net_ : instance of Module_ or ModuleList_
        Transforms the parameters produced by `matrix_handle.matrix2param_`
        -- e.g. eigenvalues, Euler angles, or whatever else that handle
        parametrizes with.

    matrix_handle : class instance
        Defines the parametrization scheme entirely, including the
        Jacobian of each direction. Specifically, it must expose two
        methods:
            `matrix2param_(matrix) -> (param, logJ)`
            `param2matrix_(param, reduce_) -> (matrix, logJ)`
    """

    def __init__(self, param_net_: Module_, *, matrix_handle):
        super().__init__()
        self.param_net_ = param_net_
        self.matrix_handle = matrix_handle

    def forward(
        self,
        x: torch.Tensor,
        log0: torch.Tensor | float = 0,
        reduce_: bool = False,
        args=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the forward transformation.

        Args:
            x: Input matrix (or, for U(1), a complex phase).
            log0: Log-Jacobian of past transformations. (Default is 0.)
            reduce_: Forwarded to `matrix_handle.param2matrix_`.
            args: Optional context, forwarded to `param_net_.forward`.

        Returns:
            The transformed matrix (or phase) and the updated log-Jacobian.
        """
        return self._kernel(
            x, is_forward=True, reduce_=reduce_, log0=log0, args=args
        )

    def reverse(
        self,
        x: torch.Tensor,
        log0: torch.Tensor | float = 0,
        reduce_: bool = False,
        args=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the reverse (inverse) transformation.

        Args:
            x: Input matrix (or, for U(1), a complex phase).
            log0: Log-Jacobian of past transformations. (Default is 0.)
            reduce_: Forwarded to `matrix_handle.param2matrix_`.
            args: Optional context, forwarded to `param_net_.reverse`.

        Returns:
            The transformed matrix (or phase) and the updated log-Jacobian.
        """
        return self._kernel(
            x, is_forward=False, reduce_=reduce_, log0=log0, args=args
        )

    def _kernel(self, matrix, *, is_forward, reduce_, log0=0, args=None):
        """Return the transformed matrix and its Jacobian.

        To this end, `matrix_handle` is used for parametrizing the input
        matrix. Then `param_net_` is used to transform the parameters.
        Finally, `matrix_handle` is used to construct a new matrix from the
        transformed parameters.
        """
        # 1. Parametrize the input matrix
        param, logJ_mat2par = self.matrix_handle.matrix2param_(matrix)

        # 2. Transform param
        args_kwargs = {} if args is None else {'args': args}
        transform = (
            self.param_net_.forward if is_forward else self.param_net_.reverse
        )
        param, logJ_par2par = transform(param, **args_kwargs)

        # 3. Construct a new matrix from the transformed parameters
        matrix, logJ_par2mat = self.matrix_handle.param2matrix_(
            param, reduce_=reduce_
        )

        # 4. Add up all log-Jacobians
        logJ = logJ_mat2par + logJ_par2par + logJ_par2mat

        return matrix, log0 + logJ

    def _hack(self, matrix, is_forward=True, reduce_=False):
        """
        Similar to the forward/reverse, but also returns intermediate objects.
        """
        # 1. Parametrize the input matrix
        param, logJ_mat2par = self.matrix_handle.matrix2param_(matrix)

        out_dict = {
            "matrix_initial": matrix,
            "param_initial": param,
            "logJ_mat2par": logJ_mat2par,
        }

        # 2. Transform param
        if is_forward:
            param, logJ_par2par = self.param_net_.forward(param)
        else:
            param, logJ_par2par = self.param_net_.reverse(param)

        out_dict.update({"param_final": param, "logJ_par2par": logJ_par2par})

        # 3. Construct a new matrix from the transformed parameters
        matrix, logJ_par2mat = self.matrix_handle.param2matrix_(
            param, reduce_=reduce_
        )
        out_dict.update({"matrix_final": matrix, "logJ_par2mat": logJ_par2mat})

        # 4. Add up all log-Jacobians
        logJ = logJ_mat2par + logJ_par2par + logJ_par2mat
        out_dict.update({"logJ": logJ})

        return out_dict
