# Copyright (c) 2021-2024 Javad Komijani

# Components of this module are mainly copied from '.matrix_handle.py'

"""This module has utilities to handle Lie groups."""


import torch
import numpy as np

from abc import abstractmethod, ABC

from ..scalar.modules_ import Tanh_

from ...lib.linalg import eigh_
from ...lib.linalg import eigu_
from ...lib.linalg import inverse_eig

from typing import Tuple


# =============================================================================
class UnitaryAlgebra2Group_(torch.nn.Module, ABC):
    r"""Maps coefficients of unitary group generators to group elements as

    .. math::

        U = e^{i \sum_a \theta_a T_a}

    where :math:`T_a` are the generators of a (special) unitary Lie aglebra.

    In addition to the group-valued matrix, this class calculates the logarithm
    of the Jacobian of the transformation.

    Using :math:`H = \sum_a \theta_a T_a`, we have

    .. math::
        J = \frac{\Delta_H}{\Delta_U}

    where :math:`\Delta_H, \Delta_U` are the conjugacy volumes corresponding
    to :math:`H, U`, respectively, as

    .. math::
        \Delta_H = \prod_{k < l} |\lambda_k - \lambda_l|^2  \\
        \Delta_U = \prod_{k < l} |e^{i\lambda_k} - e^{i\lambda_l}|^2
    """
    
    def __init__(self, coordinate_representation: bool = True):
        super().__init__()
        self.coordinate_representation = coordinate_representation
    
    def forward(self, matrix, log0 = 0):
        """Computes the unitary group matrix and log-Jacobian from the input
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
        
        # Transform coefficients to matrix if coordinate representation is True
        if self.coordinate_representation:
            matrix = self.get_matrix_representation(matrix)
        
        # 1) Eigen decomposition using `eigh_` for Hermitian matrices
        (eigvals, eigvecs), logj1 = eigh_(matrix)
        
        # 2) Map eigenvalues to the principal range
        eigvals, logj2 = self.map2principal_.forward(eigvals)
        
        # 3) Exponentiate eigenvalues
        eigvals = torch.exp(1j * eigvals)
        
        # 4) Reconstruct matrix from eigendecomposition
        matrix, logj3 = self.inverse_eign_(eigvals, eigvecs)
        
        # Return the transformed matrix and total Jacobian log
        return matrix, logj3 + logj2 + logj1 + log0
    
    def reverse(self, matrix, log0=0):
        """Inverse of the `forward` method."""

        # 4) Eigen decomposition using `eigu_` for unitary matrices
        (eigvals, eigvecs), logj1 = eigu_(matrix)
        
        # 3) Compute the angle of eigenvalues
        eigvals = torch.angle(eigvals)

        # 2) Map eigenvalues from the principal range
        eigvals, logj2 = self.map2principal_.forward(eigvals)
        
        # 1) Reconstruct matrix from eigendecomposition
        matrix, logj3 = self.inverse_eign_(eigvals, eigvecs)

        if self.coordinate_representation:
            matrix = self.extract_generators_coeffs(matrix)  # coefficients

        return matrix, logj3 + logj2 + logj1 + log0

    # Stub for the get_matrix_representation method
    def get_matrix_representation(self, coeffs):
        """Convert coefficients to an algebra-valued matrix.
        
        Parameters:
        -----------
        coeffs : torch.Tensor
            Coefficients of the Lie algebra generators.
        Returns:
        --------
            torch.Tensor: Algebra-valued matrix.
        """
        # Implement the actual transformation logic here
        pass
    
    @abstractmethod
    def extract_generators_coeffs(matrix):
        pass

    # Stub for mapping to the principal range with Jacobian log
    def map2principal_(self, eigvals) -> Tuple[torch.Tensor, float]:
        """Maps eigenvalues to principal values with log-Jacobian.
        Args:
            eigvals (torch.Tensor): Eigenvalues of the matrix.
        Returns:
            Tuple: Mapped eigenvalues and log-Jacobian.
        """
        # Implement mapping logic here
        pass
    

class Map2Principal_(Tanh_):

    def forward(self, phase):
        zeta, logj = super().forward(phase / np.pi)
        return np.pi * zeta, logj

    def reverse(self, phase):
        zeta, logj = super().reverse(phase / np.pi)
        return np.pi * zeta, logj


# =============================================================================
class SU2Algebra2Group_(UnitaryAlgebra2Group_):

    @staticmethod
    def get_matrix_representation(coeffs):

        dtype = (0j + coeffs.ravel()[0]).dtype
        matrix = torch.zeros(
                (*coeffs.shape[:-1], 2, 2), device=coeffs.device, dtype=dtype
                )

        matrix[..., 0, 0] = coeffs[..., 2]
        matrix[..., 1, 1] = - coeffs[..., 2]
        matrix[..., 1, 0] = coeffs[..., 0] + 1j * coeffs[..., 1]
        matrix[..., 0, 1] = matrix[..., 1, 0].conj()

        return matrix

    @staticmethod
    def extract_generators_coeffs(matrix):
        """It is assumed ``matrix`` is a set of 2x2 Hermitian matrices."""
        dtype = matrix.real.dtype
        coeffs = torch.zeros(
                (*matrix.shape[:-2], 3), device=matrix.device, dtype=dtype
                )

        coeffs[..., 0] = matrix[..., 1, 0].real
        coeffs[..., 1] = matrix[..., 1, 0].imag
        coeffs[..., 2] = matrix[..., 0, 0].real

        return coeffs

    @staticmethod
    def map2principal_(zero_sum_sorted_phase):
        zeta, logj = Tanh_().forward(zero_sum_sorted_phase / np.pi)
        # logj is double counted bc/ of (-theta, theta); -> divide logj by 2
        return np.pi * zeta, logj / 2

    @staticmethod
    def invert_map2principal_(zero_sum_phase):
        zeta, logj = ArcTanh_().forward(zero_sum_phase / np.pi)
        # logj is double counted bc/ of (-theta, theta); -> divide logj by 2
        return np.pi * zeta, logj / 2


# =============================================================================
class SU3Algebra2Group_(UnitaryAlgebra2Group_):

    @staticmethod
    def get_matrix_representation(coeffs):

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

        return matrix
    
    @staticmethod
    def extract_generators_coeffs(matrix):
        """It is assumed ``matrix`` is a set of 3x3 Hermitian matrices."""
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

        return coeffs

    def map2principal_(self, zero_sum_sorted_phase):
        (w, rho), logj1 = self.sortedphase2param_(zero_sum_sorted_phase)
        w, logj2 = Tanh_().forward(w)
        zero_sum_sorted_phase, logj3 = self.param2sortedphase_(w, rho)
        # print("F", logj1, logj2, logj3)
        return zero_sum_sorted_phase, logj1 + logj2 + logj3

    def invert_map2principal_(self, phase):
        order = ZeroSumOrder(phase)  # see order.(sorted_val & sorted_ind)
        zero_sum_sorted_phase = order.revert(order.sorted_val)
        (w, rho), logj1 = self.sortedphase2param_(zero_sum_sorted_phase)
        w, logj2 = ArcTanh_().forward(w)
        zero_sum_sorted_phase, logj3 = self.param2sortedphase_(w, rho)
        phase = order.revert(zero_sum_sorted_phase)
        # print("R", logj1, logj2, logj3)
        return phase, logj1 + logj2 + logj3

    @staticmethod
    def sortedphase2param_(zero_sum_sorted_phase):
        r"""Return `(w, rho)`, where `w` would be "ideally" between 0 and 1.

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
    return torch.sum(x, dim=list(range(1, x.dim())))


def calc_log_conjugacy_vol(eigvals):
    """Return log of conjugacy volume up to an additive constant."""
    
    sumlogabs2 = lambda x: 2 * torch.sum(torch.log(torch.abs(x)), dim=-1)
    
    log_vol = torch.zeros(eigvals.shape[:-1], device=eigvals.device)
    
    for k in range(eigvals.shape[-1] - 1):
        log_vol += sumlogabs2(eigvals[..., k:k+1] - eigvals[..., k+1:])
    
    return log_vol.unsqueeze(-1)  # unsqueeze to keep dimensions the same


def calc_conjugacy_vol(eigvals):
    """Return conjugacy volume up to a multiplacative constant."""
    
    prodabs2 = lambda x: torch.prod(torch.abs(x)**2, dim=-1)
    
    vol = torch.ones(eigvals.shape[:-1], device=eigvals.device)
    
    for k in range(eigvals.shape[-1] - 1):
        vol *= prodabs2(eigvals[..., k:k+1] - eigvals[..., k+1:])
    
    return vol.unsqueeze(-1)  # unsqueeze to keep dimensions the same


def calc_log_alg2grp_ratio_conjugacy_vol(eigvals):
    """Return log of ratio of conjugacy volumes of mapping a matrix from
    algebra space to group space.
    """

    sumlogabs2 = lambda x: 2 * torch.sum(torch.log(torch.abs(x)), dim=-1)

    log_vol = torch.zeros(eigvals.shape[:-1], device=eigvals.device)

    for k in range(eigvals.shape[-1] - 1):
        diff = (eigvals[..., k:k+1] - eigvals[..., k+1:]) / (2 * np.pi)
        log_vol += sumlogabs2(torch.special.sinc(diff))

    return log_vol.unsqueeze(-1)  # unsqueeze to keep dimensions the same
