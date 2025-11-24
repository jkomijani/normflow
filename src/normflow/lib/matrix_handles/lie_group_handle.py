# Copyright (c) 2021-2024 Javad Komijani

# Components of this module are mainly copied from '.matrix_handle.py'

# pylint: disable=invalid-name, arguments-differ

"""This module has utilities to handle Lie groups."""


from abc import abstractmethod, ABC

import torch
import numpy as np


from .ordering import ZeroSumOrder
from ..linalg import eigh_
from ..linalg import eigu_
from ..linalg import inverse_eigh_
from ..linalg import inverse_eign_


# =============================================================================
class UnitaryAlgebra2Group_(torch.nn.Module, ABC):
    r"""
    Maps coefficients of unitary group generators to group elements as

    .. math::

        U = e^{i \sum_a \theta_a T_a}

    where :math:`T_a` are the generators of the Lie group normalized to 1.

    In addition to the group-valued matrix, this class calculates the logarithm
    of the Jacobian of the transformation.
    """

    def __init__(
        self,
        coordinate_representation: bool = True,
        makesure_invertible: bool = True
    ):
        super().__init__()
        self.coordinate_representation = coordinate_representation
        self.makesure_invertible = makesure_invertible

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, matrix, log0=0):
        """
        Computes the unitary group matrix and log-Jacobian from the input
        matrix.

        Parameters:
        -----------
        matrix : torch.Tensor
            Either coefficients of generators or an algebra-valued matrix
            depending on `coordinate_representation`.
        log0 : torch.Tensor or float, optional
            Initial log-Jacobian value, default is 0.

        Returns:
        --------
        tuple : (torch.Tensor, torch.Tensor)
            A tuple containing:
            - The group-valued matrix.
            - The logarithm of the Jacobian of the transformation.
        """

        # 0) Transform coefficients to matrix if in coordinate representation
        if self.coordinate_representation:
            matrix = self.get_matrix_representation(matrix)

        # 1) Eigen decomposition using `eigh_` for Hermitian matrices
        (eigvals, eigvecs), logj1 = eigh_(matrix)

        # 2) Map eigenvalues to the principal cell for invertibility
        if self.makesure_invertible:
            eigvals, logj2 = self.map2principal_forward(eigvals)
        else:
            logj2 = 0

        # 3) Exponentiate eigenvalues
        eigvals = torch.exp(1j * eigvals)

        # 4) Reconstruct matrix from eigendecomposition
        matrix, logj3 = inverse_eign_(eigvals, eigvecs)

        # Return the transformed matrix and total Jacobian log
        return matrix, logj3 + logj2 + logj1 + log0

    def reverse(self, matrix, log0=0):
        """Inverse of the `forward` method."""

        # 4) Eigen decomposition using `eigu_` for unitary matrices
        (eigvals, eigvecs), logj1 = eigu_(matrix)

        # 3) Compute the angle of eigenvalues
        eigvals = torch.angle(eigvals)

        # 2) Map eigenvalues from the principal range
        if self.makesure_invertible:
            eigvals, logj2 = self.map2principal_reverse(eigvals)
        else:
            logj2 = 0

        # 1) Reconstruct matrix from eigendecomposition
        matrix, logj3 = inverse_eigh_(eigvals, eigvecs)

        # 0) Extrac coefficients from matrix if in coordinate representation
        if self.coordinate_representation:
            matrix = self.extract_generators_coeffs(matrix)  # coefficients

        # Return the transformed matrix and total Jacobian log
        return matrix, logj3 + logj2 + logj1 + log0

    @abstractmethod
    def get_matrix_representation(self, coeffs):
        """
        Convert coefficients to an algebra-valued matrix.

        Parameters:
        -----------
        coeffs : torch.Tensor
            Coefficients of the Lie algebra generators.
        Returns:
        --------
            torch.Tensor: Algebra-valued matrix.
        """
        # Implement the actual transformation logic here

    @abstractmethod
    def extract_generators_coeffs(self, matrix):
        """
        Expand matrix in basis of the Lie algebra generators.

        Parameters:
        -----------
        matrix : torch.Tensor
            Algebra-valued matrix.
        Returns:
        --------
            torch.Tensor: Coefficients of the Lie algebra generators.
        """
        # Implement the actual transformation logic here

    @staticmethod
    def map2principal_forward(phase):
        """Maps phase to the principal range with log-Jacobian."""
        zeta, logj = tanh_(phase / np.pi)
        return np.pi * zeta, logj

    @staticmethod
    def map2principal_reverse(phase):
        """Reverse mapping of phase to principal range with log-Jacobian."""
        zeta, logj = atanh_(phase / np.pi)
        return np.pi * zeta, logj


def tanh_(x):
    """
    Applies a hyperbolic tangent transformation to the input `x` and returns
    both the transformed output and the log-Jacobian of transformation.
    """
    # Apply the tanh transformation
    transformed_x = torch.tanh(x)

    # Compute log-Jacobian (summed over non-batched axes)
    logj = -2 * sum_density(torch.log(torch.cosh(x)))

    return transformed_x, logj


def atanh_(x):
    """
    Applies a hyperbolic arc-tangent transformation to the input `x` and
    returns both the transformed output and the log-Jacobian of transformation.
    """

    # Apply the arc-tanh transformation
    transformed_x = torch.atanh(x)

    # Compute log-Jacobian (summed over non-batched axes)
    logj = 2 * sum_density(torch.log(torch.cosh(transformed_x)))

    return transformed_x, logj


# =============================================================================
class SU2Algebra2Group_(UnitaryAlgebra2Group_):
    r"""
    Maps coefficients of unitary group generators to group elements as

    .. math::

        U = e^{i \sum_a \theta_a T_a}

    where :math:`T_a` are the generators of SU(2) group normalized to 1.
    """

    @staticmethod
    def get_matrix_representation(coeffs):
        """
        Convert coefficients to an algebra-valued matrix.

        Parameters:
        -----------
        coeffs : torch.Tensor
            Coefficients of the Lie algebra generators with three components on
            the outermost axis.
        Returns:
        --------
            torch.Tensor: Algebra-valued `2 x 2` matrix on the outermost axes.
        """
        dtype = (0j + coeffs.ravel()[0]).dtype
        matrix = torch.zeros(
                (*coeffs.shape[:-1], 2, 2), device=coeffs.device, dtype=dtype
                )

        matrix[..., 0, 0] = coeffs[..., 2]
        matrix[..., 1, 1] = - coeffs[..., 2]
        matrix[..., 1, 0] = coeffs[..., 0] + 1j * coeffs[..., 1]
        matrix[..., 0, 1] = matrix[..., 1, 0].conj()

        return matrix / 2 ** 0.5

    @staticmethod
    def extract_generators_coeffs(matrix):
        """
        Expand matrix in basis of the Lie algebra generators.

        Parameters:
        -----------
        matrix : torch.Tensor
            Algebra-valued `2 x 2` matrix on the outermost axes.
        Returns:
        --------
            torch.Tensor: Coefficients of the Lie algebra generators.
        """
        dtype = matrix.real.dtype
        coeffs = torch.zeros(
                (*matrix.shape[:-2], 3), device=matrix.device, dtype=dtype
                )

        coeffs[..., 0] = matrix[..., 1, 0].real
        coeffs[..., 1] = matrix[..., 1, 0].imag
        coeffs[..., 2] = matrix[..., 0, 0].real

        return coeffs * 2**0.5

    @staticmethod
    def map2principal_forward(zero_sum_sorted_phase):
        """Maps phase to the principal range with log-Jacobian."""
        zeta, logj = tanh_(zero_sum_sorted_phase / np.pi)
        # logj is double-counted bc/ of (-theta, theta); -> divide logj by 2
        return np.pi * zeta, logj / 2

    @staticmethod
    def map2principal_reverse(zero_sum_phase):
        """Reverse mapping of phase to principal range with log-Jacobian."""
        zeta, logj = atanh_(zero_sum_phase / np.pi)
        # logj is double-counted bc/ of (-theta, theta); -> divide logj by 2
        return np.pi * zeta, logj / 2


# =============================================================================
class SU3Algebra2Group_(UnitaryAlgebra2Group_):
    r"""
    Maps coefficients of unitary group generators to group elements as

    .. math::

        U = e^{i \sum_a \theta_a T_a}

    where :math:`T_a` are the generators of SU(2) group normalized to 1.
    """

    @staticmethod
    def get_matrix_representation(coeffs):
        """
        Convert coefficients to an algebra-valued matrix.

        Parameters:
        -----------
        coeffs : torch.Tensor
            Coefficients of the Lie algebra generators with eight components on
            the outermost axis.
        Returns:
        --------
            torch.Tensor: Algebra-valued `3 x 3` matrix on the outermost axes.
        """
        dtype = (0j + coeffs.ravel()[0]).dtype
        matrix = torch.zeros(
                (*coeffs.shape[:-1], 3, 3), device=coeffs.device, dtype=dtype
                )

        # diagonal
        matrix[..., 0, 0] = coeffs[..., 2] + coeffs[..., 7] / 3**0.5
        matrix[..., 1, 1] = - coeffs[..., 2] + coeffs[..., 7] / 3**0.5
        matrix[..., 2, 2] = coeffs[..., 7] * (-2 / 3**0.5)
        # off diagonal
        matrix[..., 1, 0] = coeffs[..., 0] + 1j * coeffs[..., 1]
        matrix[..., 2, 0] = coeffs[..., 3] + 1j * coeffs[..., 4]
        matrix[..., 2, 1] = coeffs[..., 5] + 1j * coeffs[..., 6]
        matrix[..., 0, 1] = matrix[..., 1, 0].conj()
        matrix[..., 0, 2] = matrix[..., 2, 0].conj()
        matrix[..., 1, 2] = matrix[..., 2, 1].conj()

        return matrix / 2**0.5

    @staticmethod
    def extract_generators_coeffs(matrix):
        """
        Expand matrix in basis of the Lie algebra generators.

        Parameters:
        -----------
        matrix : torch.Tensor
            Algebra-valued `3 x 3` matrix on the outermost axes.
        Returns:
        --------
            torch.Tensor: Coefficients of the Lie algebra generators.
        """
        dtype = matrix.real.dtype
        coeffs = torch.zeros(
                (*matrix.shape[:-2], 8), device=matrix.device, dtype=dtype
                )

        # diagonal
        coeffs[..., 2] = (matrix[..., 0, 0].real - matrix[..., 1, 1].real) / 2
        coeffs[..., 7] = matrix[..., 2, 2].real / (-2 / 3**0.5)
        # off diagonal
        coeffs[..., 0] = matrix[..., 1, 0].real
        coeffs[..., 1] = matrix[..., 1, 0].imag
        coeffs[..., 3] = matrix[..., 2, 0].real
        coeffs[..., 4] = matrix[..., 2, 0].imag
        coeffs[..., 5] = matrix[..., 2, 1].real
        coeffs[..., 6] = matrix[..., 2, 1].imag

        return coeffs * 2**0.5

    def map2principal_forward(self, zero_sum_sorted_phase):
        """
        Maps the input phase to the principal range.

        Parameters:
        -----------
        zero_sum_sorted_phase : torch.Tensor
            A tensor of sorted phases where the phases sum to zero.

        Returns:
        --------
        tuple:
            - zero_sum_sorted_phase (torch.Tensor):
                The transformed phase in the principal range.
            - log_jacobian (torch.Tensor):
                Sum of log-Jacobians from each transformation step.
        """

        # Step 1: Transform input phase to parameters `(w, rho)`
        (w, rho), logj1 = self.sortedphase2param_(zero_sum_sorted_phase)

        # Step 2: Apply tanh transformation to `w`
        w, logj2 = tanh_(w)

        # Step 3: Map `(w, rho)` back to sorted phase
        zero_sum_sorted_phase, logj3 = self.param2sortedphase_(w, rho)

        return zero_sum_sorted_phase, logj1 + logj2 + logj3

    def map2principal_reverse(self, phase):
        """Reverse mapping of phase to principal range with log-Jacobian."""

        # Preconditioning: sort the phase and make the sum zero.
        order = ZeroSumOrder(phase)  # see order.(sorted_val & sorted_ind)
        zero_sum_sorted_phase = order.sorted_val

        # Reverse step 3: Transform input phase to parameters `(w, rho)`
        (w, rho), logj3 = self.sortedphase2param_(zero_sum_sorted_phase)

        # Reverse Step 2: Apply atanh transformation to `w`
        w, logj2 = atanh_(w)

        # Reverse step 1: Transform `(w, rho)` back to sorted phase.
        zero_sum_sorted_phase, logj1 = self.param2sortedphase_(w, rho)

        # Undo Preconditioning: sort the phase and make the sum zero.
        phase = order.revert(zero_sum_sorted_phase)

        return phase, logj1 + logj2 + logj3

    @staticmethod
    def sortedphase2param_(zero_sum_sorted_phase):
        r"""
        Return `(w, rho)`, where `w` would be "ideally" between 0 and 1.

        .. math::

            w = \theta \cos (\phi) / \pi

        and `rho` that is independent of :math:`theta`.
        """
        x, y, z = zero_sum_sorted_phase.split((1, 1, 1), dim=-1)  # x <= y <= z
        w = (z - x) / (2 * np.pi)  # w \in [0, 1]
        w[w == 0] = 1e-16
        rho = y / w
        logj = -sum_density(torch.log(w))
        # c = np.log(8/3 * pi**2), additive constant to log(w), but we drop it
        return (w, rho), logj

    @staticmethod
    def param2sortedphase_(w, rho):
        """Inverse of sortedphase2param_()"""
        y = w * rho
        z = -y / 2 + w * np.pi
        x = -y / 2 - w * np.pi
        logj = sum_density(torch.log(w))
        # c = np.log(8/3 * pi**2), additive constant to log(w), but we drop it
        return torch.cat((x, y, z), dim=-1), logj


# =============================================================================
def sum_density(x):
    """Compute the sum over all, but the batch, axes."""
    return torch.sum(x, dim=list(range(1, x.dim())))
