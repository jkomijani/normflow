# Copyright (c) 2021-2022 Javad Komijani

"""This module contains new neural networks for transforming matrices.

The classes defined here are subclasses of `Module_`, and like it, the trailing
underscore implies that the associated forward and reverse methods handle the
Jacobians of the transformation.
"""

# pylint: disable=invalid-name, relative-beyond-top-level, too-many-arguments

from .._core import Module_


# =============================================================================
class MatrixModule_(Module_):
    """A module for transforming matrices.

    All matrix-specific logic (parametrization, reconstruction, and the
    associated Jacobians) is delegated to `matrix_handle`; this class itself
    is agnostic to the group. In particular, it also works for U(1) theory:
    pass `matrix_handle=U1Parametrizer()` (see
    `normflow.lib.matrix_handles`), in which case `x` is a plain complex
    phase rather than a matrix.

    Parameters
    ----------
    param_net_: instance of Module_ or ModuleList_
        to change the parameters corresponding to the matrices, e.g.,
        eigenvaleus of the matrices.

    matrix_handle: class instance
        for parametrization of the matrices. For more information on how it is
        used, see `self._kernel`.
    """

    def __init__(self, param_net_, *, matrix_handle):
        super().__init__()
        self.param_net_ = param_net_
        self.matrix_handle = matrix_handle

    def forward(self, x, log0=0, reduce_=False, args=None):
        """Apply forward transformation."""
        return self._kernel(
            x, is_forward=True, reduce_=reduce_, log0=log0, args=args
        )

    def reverse(self, x, log0=0, reduce_=False, args=None):
        """Apply reverse (inverse) transformation."""
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
        if is_forward:
            if args is None:
                param, logJ_par2par = self.param_net_.forward(param)
            else:
                param, logJ_par2par = self.param_net_.forward(param, args=args)
        else:
            if args is None:
                param, logJ_par2par = self.param_net_.reverse(param)
            else:
                param, logJ_par2par = self.param_net_.reverse(param, args=args)

        # 3. Construct a new matrix from the transformed parameters
        matrix, logJ_par2mat = self.matrix_handle.param2matrix_(
            param, reduce_=reduce_
        )

        # 4. Add up all log-Jacobians
        logJ = logJ_mat2par + logJ_par2par + logJ_par2mat

        return matrix, log0 + logJ

    def _hack(self, matrix, is_forward=True, reduce_=False):
        """Similar to the forward/reverse methods, but returns intermediate
        parts too.
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

        # 6. Add up all log-Jacobians
        logJ = logJ_mat2par + logJ_par2par + logJ_par2mat
        out_dict.update({"logJ": logJ})

        return out_dict
