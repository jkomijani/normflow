# Copyright (c) 2024-2025 Javad Komijani

import torch

from lattice_ml.linalg import eyes_like
from lattice_ml.linalg import kronecker_product

from lattice_ml.functions import matrix_exp1jh_and_jacobian
from lattice_ml.functions import inverse_eign_and_jacobian
from lattice_ml.functions import commutator_and_jacobian


# =============================================================================
class UnitaryFlow_:
    r"""For a transformation of type

    .. math:

        V = f(U, args) U;

    where :math:`U` and :math:`F = f(U, args)` are unitary. In addition to the
    given transformation, the Jacobian of transformation is also calculated and
    returned.

    In the reverse mode, the reverse of the function is evaluated iteratively.

    Parameters
    ----------
    func : (partial) function
        Takes :math:`U` as input and returns :math:`f(U, args)` along with the
        Jacobian of transformation wrt to :math:`\int d\Gamma`, where
        :math:`U^\dagger dU`.
    """

    jacobian_mode = 'Gamma'  # can be changed to `Omega` if needed.
    return_logdet = True  # return log(|det(J)|).

    def __init__(self, func, n_steps=1, reverse_mode_iter=10):
        # e.g., func = modal2antihermitian2unitary
        self.func = func
        self.n_steps = n_steps
        self.reverse_mode_iter = reverse_mode_iter

    def __call__(self, matrix, **func_kwargs):
        return self.forward(matrix, **func_kwargs)

    def forward(self, matrix, **func_kwargs):
        # This can be used ONLY with `return_logdet = True`
        full_logJ = 0
        for _ in range(self.n_steps):
            matrix, logJ = self.one_step_forward(matrix, **func_kwargs)
            full_logJ += logJ
        return matrix, full_logJ

    def reverse(self, matrix, **func_kwargs):
        # This can be used ONLY with `return_logdet = True`
        full_logJ = 0
        for _ in range(self.n_steps):
            matrix, logJ = self.one_step_reverse(matrix, **func_kwargs)
            full_logJ += logJ
        return matrix, full_logJ

    def one_step_forward(self, u_matrix, **func_kwargs):
        f_matrix, f_jacobian = self.func(u_matrix, **func_kwargs)
        v_matrix = f_matrix @ u_matrix
        jacobian = self.calc_jacobian_matrix(u_matrix, v_matrix, f_jacobian)
        if self.return_logdet:
            return v_matrix, self.calc_logdet(jacobian)
        else:
            return v_matrix, jacobian

    def one_step_reverse(self, v_matrix, **func_kwargs):
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
        eye = eyes_like(f_jacobian)
        mat = kronecker_product(v_matrix.adjoint(), u_matrix.transpose(-2, -1))
        jac = eye + mat @ f_jacobian
        if self.jacobian_mode == 'Omega':
            eye = eyes_like(u_matrix)
            jac = kronecker_product(v_matrix, eye) @ jac
        return jac

    @staticmethod
    def calc_logdet(jacobian):
        return torch.log(torch.linalg.det(jacobian).abs())


# =============================================================================
def modal2antihermitian2unitary(omega, *, lambda_, sigma, tau=1, mode='Gamma'):
    r"""Return :math:`\exp(-\tau [H, \Sigma])` and its Jacobian matrix,
    where :math:`H = \Omega \Lambda \Omega^\dagger` and :math:`\Sigma`
    are Hermitian matrices.

    Remark: `tau` must be a number or scalar with respect to the last two axes.
    """
    matrix, jac1 = inverse_eign_and_jacobian(lambda_, omega, mode=mode)
    matrix, jac2 = commutator_and_jacobian(matrix, sigma)
    matrix, jac3 = matrix_exp1jh_and_jacobian((1j * tau) * matrix)
    jac = (1j * tau) * (jac3 @ jac2 @ jac1)
    return matrix, jac
