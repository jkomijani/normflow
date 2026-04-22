# Copyright (c) 2024-2026 Javad Komijani

"""
Unitary flows and modal parameterizations.

This module provides tools to construct differentiable transformations on
the unitary group U(n) based on Lie algebra updates and spectral (modal)
representations.
"""

# pylint: disable=invalid-name

import torch

from lattice_ml.functions import eyes_like
from lattice_ml.functions import kronecker_product
from lattice_ml.functions import matrix_exp1jh_and_jacobian
from lattice_ml.functions import inverse_eign_and_jacobian
from lattice_ml.functions import commutator_and_jacobian

__all__ = ["UnitaryFlow_", "transform_modal2antihermitian2unitary"]


# =============================================================================
class UnitaryFlow_:
    """
    Iterative unitary flow of the form
        V = f(U, args) @ U,

    where both U and F = f(U, args) are unitary matrices (or SU(n) matrices).
    The transformation is applied repeatedly and its Jacobian is accumulated
    across steps.

    The map defines a flow on the unitary group, with local updates given by
    left multiplication. In addition to the forward transformation, an
    approximate inverse is computed via fixed-point iteration.

    Parameters
    ----------
    func : callable
        Function implementing the update:
            F, J = func(U, **args)
        where F = f(U, args) is unitary and J is the Jacobian of f with respect
        to the chosen parametrization (e.g. Γ with U† dU).
        For instance, `func = transform_modal2antihermitian2unitary`.
    n_steps : int, optional
        Number of flow steps to apply (default: 1).
    reverse_mode_iter : int, optional
        Number of fixed-point iterations used to approximate the inverse map.

    Attributes
    ----------
    jacobian_mode : str
        Choice of Jacobian representation ('Gamma' or 'Omega').
    return_logdet : bool
        If True, return log|det(J)| instead of the full Jacobian matrix.
    """

    jacobian_mode = 'Gamma'  # can be changed to `Omega` if needed.
    return_logdet = True  # return log(|det(J)|).

    def __init__(self, func, n_steps=1, reverse_mode_iter=10):
        self.func = func
        self.n_steps = n_steps
        self.reverse_mode_iter = reverse_mode_iter

    def __call__(self, matrix, **func_kwargs):
        """Alias for forward."""
        return self.forward(matrix, **func_kwargs)

    def forward(self, matrix, **func_kwargs):
        """
        Apply the forward flow.

        Returns the transformed matrix and accumulated log-determinant.
        """
        # This can be used ONLY with `return_logdet = True`
        full_logJ = 0
        for _ in range(self.n_steps):
            matrix, logJ = self.one_step_forward(matrix, **func_kwargs)
            full_logJ += logJ
        return matrix, full_logJ

    def reverse(self, matrix, **func_kwargs):
        """
        Apply the approximate inverse flow.

        Returns the recovered matrix and accumulated log-determinant.
        """
        # This can be used ONLY with `return_logdet = True`
        full_logJ = 0
        for _ in range(self.n_steps):
            matrix, logJ = self.one_step_reverse(matrix, **func_kwargs)
            full_logJ += logJ
        return matrix, full_logJ

    def one_step_forward(self, u_matrix, **func_kwargs):
        """
        Apply a single forward step: V = F(U) @ U.

        Returns the updated matrix and Jacobian (or its log-determinant).
        """
        f_matrix, f_jacobian = self.func(u_matrix, **func_kwargs)
        v_matrix = f_matrix @ u_matrix
        jacobian = self.calc_jacobian_matrix(u_matrix, v_matrix, f_jacobian)
        if self.return_logdet:
            return v_matrix, self.calc_logdet(jacobian)
        else:
            return v_matrix, jacobian

    def one_step_reverse(self, v_matrix, **func_kwargs):
        """
        Apply a single reverse step via fixed-point iteration.

        Returns the approximate inverse and Jacobian (or its log-determinant).
        """
        u_tentative = v_matrix
        for _ in range(self.reverse_mode_iter):
            f_tentative, f_jacobian = self.func(u_tentative, **func_kwargs)
            u_tentative = f_tentative.adjoint() @ v_matrix
        jacobian = self.calc_jacobian_matrix(u_tentative, v_matrix, f_jacobian)
        if self.return_logdet:
            return u_tentative, -self.calc_logdet(jacobian)
        else:
            return u_tentative, torch.linalg.inv(jacobian)

    def calc_jacobian_matrix(self, u_matrix, v_matrix, f_jacobian):
        """
        Construct the Jacobian of the full transformation using the chain rule.
        """
        eye = eyes_like(f_jacobian)
        mat = kronecker_product(v_matrix.adjoint(), u_matrix.transpose(-2, -1))
        jac = eye + mat @ f_jacobian
        if self.jacobian_mode == 'Omega':
            eye = eyes_like(u_matrix)
            jac = kronecker_product(v_matrix, eye) @ jac
        return jac

    @staticmethod
    def calc_logdet(jacobian):
        """Compute log |det(J)| of the Jacobian."""
        return torch.log(torch.linalg.det(jacobian).abs())


# =============================================================================
def transform_modal2antihermitian2unitary(
    Omega: torch.Tensor,
    *,
    diag_Lambda: torch.Tensor,
    Sigma: torch.Tensor,
    tau: float = 1,
    mode: str = 'Gamma'
):
    """
    Compute the unitary matrix
        U = exp(-τ [H, Σ])
    together with its Jacobian, where H is Hermitian reconstructed from its
    real spectral matrix Λ and modal matrix Ω as:
        H = Ω Λ Ω†.

    Here, Σ is assumed Hermitian (up to an additive purely imaginary scalar
    multiple of the identity, which does not affect the commutator).

    The mapping proceeds in three steps:
        (Ω, Λ) → H → [H, Σ] → U = exp(-τ [H, Σ])

    and the total Jacobian is obtained via the chain rule.

    Args:
        Omega (torch.Tensor): Eigenvector matrix Ω (..., n, n).
        diag_Lambda (torch.Tensor): Eigenvalues Λ (..., n).
        Sigma (torch.Tensor): Hermitian-like matrix Σ (..., n, n).
        tau (float): Scalar step size τ.
        mode (str): Mode flag passed to eigen reconstruction.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - U: Unitary matrix exp(-τ [H, Σ]) (..., n, n)
            - J: Jacobian of U w.r.t. the modal parameters

    Notes:
        - The commutator [H, Σ] is skew-Hermitian, so U is unitary.
        - τ must be a scalar with respect to the last two dimensions.
    """
    # Step 1: reconstruct H = Ω Λ Ω† and its Jacobian
    H, jac1 = inverse_eign_and_jacobian(diag_Lambda, Omega, mode=mode)

    # Step 2: compute commutator C = [H, Σ] and its Jacobian
    C, jac2 = commutator_and_jacobian(H, Sigma)

    # Step 3: exponentiate to obtain U = exp(i (i τ C)) and its Jacobian
    U, jac3 = matrix_exp1jh_and_jacobian((1j * tau) * C)

    # Chain rule: combine Jacobians of the three transformations
    jac = (1j * tau) * (jac3 @ jac2 @ jac1)

    return U, jac
