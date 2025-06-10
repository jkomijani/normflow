# Copyright (c) 2025 Javad Komijani

r"""
Wilson Modal Flow Module for Continuous Normalizing Flows on Unitary Matrices.

This module implements a flow-based transformation inspired by Wilson's action
in lattice gauge theory. The core components are:

Mathematical Formulation:

.. math::

    dX / dt = - a [X \Lambda X^\dagger, \Sigma] X

where:
    - X is a unitary matrix (or a batch of them), or a combination of unitary
        and zero matrices when a mask is used.
    - Lambda is a diagonal matrix of real eigenvalues.
    - Sigma is a Hermitian matrix.
    - [A, B] = AB - BA is the matrix commutator.
    - The overall coefficient is `a = tau / (2 n_c^2)`.
    - tau is a learnable scalar controlling flow speed.
    - `n_c` is the number of colors of the gauge theory (dimension of X).


The flow supports efficient computation of the log-determinant of the Jacobian
for use in continuous normalizing flows.
"""

import torch

from lattice_ml.integrate import AdjLieODEFlow_
from lattice_ml.integrate import LieODEFlow_
from lattice_ml.integrate import AdjLieModule

from lattice_ml.linalg import inverse_eign  # reconstructs from eigenbasis

from .._core import Module_

Tensor = torch.Tensor


# =============================================================================
class WilsonModalFlow_(Module_):  # pylint: disable=invalid-name
    """
    Implements a masked Wilson flow applied on the modal matrix, using an
    adjoint Lie ODE solver. Supports forward and reverse passes.
    """

    def __init__(
        self,
        t_span=(0, 1),
        step_size=0.1,
        tau_par=None,
        mask=None,
        use_adjoint_method=True,
        **solver_kwargs
    ):
        """
        Initialize the WilsonModalFlow_ module.

        Args:
            t_span (tuple): Time span for ODE integration.
            step_size (float): Integration step size.
            tau_par (Tensor or None): Initial tau parameter; defaults to 0.
            mask (optional): Mask object for splitting and recombining tensors.
            use_adjoint_method (optional): If True uses the adjoint method.
            **solver_kwargs: Additional arguments for the ODE solver.
        """

        super().__init__()

        flow_class = AdjLieODEFlow_ if use_adjoint_method else LieODEFlow_

        self.flow_ = flow_class(
            WilsonModalFlowDynamics(),
            t_span=t_span,
            step_size=step_size,
            **solver_kwargs
        )

        self.mask = mask

        if tau_par is None:
            self.tau_par = torch.nn.Parameter(torch.zeros(1))
        else:
            self.tau_par = tau_par

    def forward(self, eigvecs, *, eigangs, staples_object):
        """
        Applies the forward Wilson flow to the eigenvectors.

        Args:
            eigvecs (Tensor): Eigenvector input.
            eigangs (Tensor): Eigenvalue angles.
            staples_object: Object containing SVD decomposition (Sigma, angle).

        Returns:
            Tuple[Tensor, Tensor]: Transformed eigenvectors and log Jacobian.
        """
        alpha = staples_object.svd_.rdet_angle.unsqueeze(-1)
        lambda_vector = torch.cos(eigangs + alpha) + 0j
        sigma_matrix = staples_object.svd_.Sigma
        tau = self.tau_net()
        args = (lambda_vector, sigma_matrix, tau)
        if self.mask is None:
            eigvecs, logj = self.flow_.forward(eigvecs, args=args)
        else:
            # Apply flow only to masked portion of the data
            x_0, x_1 = self.mask.split(eigvecs)
            x_0, logj = self.flow_.forward(x_0, args=args)
            eigvecs = self.mask.cat(x_0, x_1)
        return eigvecs, logj

    def reverse(self, eigvecs, *, eigangs, staples_object):
        """
        Applies the reverse Wilson flow to the eigenvectors.

        Args:
            eigvecs (Tensor): Transformed eigenvectors.
            eigangs (Tensor): Eigenvalue angles.
            staples_object: Object containing SVD decomposition (Sigma, angle).

        Returns:
            Tuple[Tensor, Tensor]: Recovered eigenvectors and log Jacobian.
        """
        alpha = staples_object.svd_.rdet_angle.unsqueeze(-1)
        lambda_vector = torch.cos(eigangs + alpha) + 0j
        sigma_matrix = staples_object.svd_.Sigma
        tau = self.tau_net()
        args = (lambda_vector, sigma_matrix, tau)
        if self.mask is None:
            eigvecs, logj = self.flow_.reverse(eigvecs, args=args)
        else:
            # Apply reverse flow only to masked portion of the data
            x_0, x_1 = self.mask.split(eigvecs)
            x_0, logj = self.flow_.reverse(x_0, args=args)
            eigvecs = self.mask.cat(x_0, x_1)
        return eigvecs, logj

    def tau_net(self):
        """
        Computes a softplus activation of the learned tau parameter minus 1.

        Returns:
            Tensor: Positive-valued tau used in flow dynamics.
        """
        return torch.nn.functional.softplus(self.tau_par - 1)


# =============================================================================
class WilsonModalFlowDynamics(AdjLieModule):
    r"""
    Implements the modal-matrix flow dynamics derived from the Wilson action.

    The dynamics are defined using commutator flows:

    .. math::

        dX / dt = - a [X \Lambda X^\dagger, \Sigma] X

    where:
        - X is a Unitary matrix, or a combination of unitary and zero matrices
          when a mask is used.
        - Lambda is a diagonal matrix of real eigenvalues.
        - Sigma is a Hermitian matrix.
        - [A, B] = AB - BA is the matrix commutator.
        - The overall coefficient is `a = tau / (2 n_c^2)`.
        - tau is a learnable scalar controlling flow speed.
        - `n_c` is the number of colors of the gauge theory (dimension of X).

    The above expression is equivalent to:

    .. math::

        dX / dt = - a (X \Lambda X^\dagger \Sigma - \text{h.c.}) X

    where h.c. denotes the Hermitian conjugate (i.e., conjugate transpose)
    of the preceding term. This form is preferred because automatic
    differentiation (AD) with complex matrices can be subtle.
    In particular, computing the Jacobian using the commutator form may lead to
    a factor-of-two discrepancy due to how AD handles complex-valued functions.
    By contrast, the right-hand side of the expression above is explicitly
    anti-Hermitian, which removes ambiguity in derivative calculations when
    using AD.

    The log-Jacobian rate used in continuous normalizing flows is:

    .. math::

        \frac{d}{dt} \log |det J| = - 2 a (
            n_c Tr(X \Lambda X^\dagger \Sigma) - Tr(\Sigma) Tr(\Lambda)
        )
    """

    return_logj_density = False

    def algebra_dynamics(self, t, x, lambda_vector, sigma_matrix, tau):
        """
        Computes the Lie algebra dynamics at time t.

        Args:
            t (float): Time parameter (not used but included for API).
            x (Tensor): The modal matrix or zero matrix, or a combination.
            lambda_vector (Tensor): The eigenvalue vector.
            sigma_matrix (Tensor): The Hermitian matrix influencing the flow.
            tau (float): The scaling factor.

        Note:
            It is the user's responsibility to ensure that `x` is unitary,
            `lambda_vector` contains real values, and `sigma_matrix` is
            Hermitian.

        Returns:
            Tensor: Time derivative of x based on commutator evolution.
        """
        # Flow is governed by a commutator of transformed eigenbasis and noise
        n_c = x.shape[-1]
        coeff = - tau / (2 * n_c**2)
        product = inverse_eign(lambda_vector, x) @ sigma_matrix
        return coeff * (product - product.adjoint())

    def calc_logj_rate(self, t, x, lambda_vector, sigma_matrix, tau):
        """
        Computes the rate of change of the log-Jacobian determinant.

        Args:
            t (float): Time parameter (not used but included for API).
            x (Tensor): The modal matrix or zero matrix, or a combination.
            lambda_vector (Tensor): The eigenvalue vector.
            sigma_matrix (Tensor): The Hermitian matrix influencing the flow.
            tau (float): The scaling factor.

        Returns:
            Tensor: Scalar log-Jacobian rate of change.
        """
        # First trace: transformed inverse eigenvalue matrix product with sigma
        trace_1 = calc_trace(inverse_eign(lambda_vector, x) @ sigma_matrix)

        # Second trace: trace of sigma times the sum over lambda vector
        trace_2 = calc_trace(sigma_matrix) * torch.sum(lambda_vector, dim=-1)

        n_c = x.shape[-1]

        # Following line is not needed if x is truely a modal matrix, but in
        # practice because of the use of masks, x might be a zero matrix too.
        trace_2 *= calc_trace(x @ x.adjoint()).real / n_c

        if x.ndim > 3 and not self.return_logj_density:
            dim = tuple(range(1, x.ndim - 2))
            # 0: batch axis, -1 & -2: matrix
            trace_1 = torch.sum(trace_1, dim=dim)
            trace_2 = torch.sum(trace_2, dim=dim)

        coeff = - tau / (2 * n_c**2)

        # Total log-Jacobian rate formula
        return (2 * coeff) * (n_c * trace_1 - trace_2).real


# =============================================================================
def commutator(mat1: Tensor, mat2: Tensor) -> Tensor:
    """Returns the commutator of two square matrices `[A, B]`."""
    return mat1 @ mat2 - mat2 @ mat1


def calc_trace(x: Tensor) -> Tensor:
    """Returns the trace of a square matrix."""
    return torch.sum(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)
