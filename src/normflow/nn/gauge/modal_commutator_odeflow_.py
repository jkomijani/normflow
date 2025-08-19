# Copyright (c) 2025 Javad Komijani

r"""
This module implements commutator-based Lie algebra flows on unitary (modal)
matrices, inspired by Wilson's action in lattice gauge theory. These flows
operate in the space of eigenvector matrices and leverage Lie group and
Lie algebra structures for efficient transformations.

Overview
--------

The flow acts on the modal matrix `X` (containing eigenvectors) via an ODE of
the form:

.. math::

    \frac{dX}{dt} = -a [X \Lambda X^\dagger, \Sigma] X

where:
    - `X` is unitary (but can be masked or zeroed),
    - `Lambda` is a (traceless) diagonal matrix of real eigenvalues.
    - `Sigma` is a (traceless) Hermitian matrix.
    - `[A, B] = AB - BA` is the matrix commutator.
    - `a` is a scaling factor.

Classes
-------

- ModalCommutatorDynamics:
    Defines only the Lie algebra-valued dynamics (i.e., the right-hand side of
    the ODE).

    For computational efficieny, this class assumes that `Lambda` and `Sigma`
    are traceless, which is valid since their trace components drop out
    of the commutator.

    This class inherits from `AdjLieModule`, and overrides key methods of the
    base class for log-Jacobian rate and adjoint-method based backpropagation
    with analytic closed-form formula, improving performance compared to the
    base class's automatic differentiation defaults.

    Notably, although `X` is supposed to be unitary, all formulas used for
    Jacobian calculation and backpropagation remain valid even when `X` is
    zeroed out. This allows the use of this class in combination with masks
    over `X`.

    Note: This class does *not* perform ODE integration. To execute the flow,
    one can wrap it inside an ODE solver, e.g.:
        >>> AdjLieODEFlow_(ModalCommutatorDynamics(), **solver_kwargs)

- ModalCommutatorFlow_:
    A high-level module that wraps `ModalCommutatorDynamics` inside a Lie group
    ODE solver (`AdjLieODEFlow_` or `LieODEFlow_`) to perform time integration.

    This class is designed for use with lattice QCD simulation data, where
    the inputs consist of eigenvector matrices and corresponding gauge-derived
    quantities.

    Features include:
        - Forward and reverse integration of the flow,
        - Interfaces for SVD-based flow of gauge links,
        - Optional masked application to partial subsets of the modal matrix,
        - Handling of tracelessness assumptions for `Lambda` and `Sigma`,
        - Learnable flow strength via a `tau` parameter.

- SlowModalCommutatorDynamics:
    A reference (non-optimized) version of the flow dynamics using automatic
    differentiation for all computations, including Jacobian and gradients.
    It inherits from `AdjLieModule` without overriding any methods.

    This class is intended strictly for:
        - Sanity checks,
        - Gradient validation,
        - Benchmarking performance against `ModalCommutatorDynamics`.

    It should not be used in production models due to its significantly slower
    performance.

Architecture and Performance Notes
----------------------------------

- The base class `AdjLieModule` defines the core interface and includes
  default implementations of methods such as calc_logj_rate, aug_reverse,
  and calc_grad_params_rate using automatic differentiation.

- ModalCommutatorDynamics overrides these methods with analytic formulas,
  significantly speeding up both training and inference.

- ModalCommutatorFlow_ manages the assumption that Sigma and Lambda are
  traceless, ensuring this condition is handled appropriately when performing
  flows and computing log-Jacobians.

- Masking support in ModalCommutatorFlow_ enables flexible splitting and
  recombining of modal matrices, facilitating efficient computations in large
  lattice systems.
"""

# pylint: disable=relative-beyond-top-level, arguments-differ, too-many-locals
# pylint: disable=too-many-arguments, too-many-positional-arguments


import torch

from lattice_ml.integrate import AdjLieODEFlow_
from lattice_ml.integrate import LieODEFlow_
from lattice_ml.integrate import AdjLieModule
from lattice_ml.integrate import TupleVar

from lattice_ml.linalg import inverse_eigh  # reconstructs from eigenbasis

from .._core import Module_

Tensor = torch.Tensor


__all__ = [
    "ModalCommutatorFlow_",
    "ModalCommutatorDynamics",
    "SlowModalCommutatorDynamics"
]


# =============================================================================
class ModalCommutatorFlow_(Module_):  # pylint: disable=invalid-name
    """
    High-level module for applying a masked commutator-based flow to modal
    (eigenvector) matrices using a Lie group ODE solver.
    The commutator-based ODE derived from the Wilson gauge action using SVD.

    This class wraps `ModalCommutatorDynamics` inside either an adjoint or
    standard ODE solver (`AdjLieODEFlow_` or `LieODEFlow_`), and is designed
    for use in lattice QCD pipelines where eigenvectors and eigenvalues are
    extracted from gauge configurations.

    Features:
        - Applies commutator flow to unitary eigenvector matrices.
        - Supports **masked application**, enabling selective transformation of
          sub-regions of the input.
        - Contains a learnable scalar `tau_par`, which is passed through
          `tau_net()` to produce a flow-scaling parameter `tau`.
        - Supports both **forward** and **reverse** integration over a time
          interval `t_span`, enabling invertible transformations.
    """

    def __init__(
        self,
        t_span: tuple = (0, 1),
        step_size: float = 0.1,
        tau_par: torch.nn.Parameter | torch.Tensor | None = None,
        mask=None,
        use_adjoint_method: bool = True,
        **solver_kwargs
    ):
        """
        Initializes the ModalCommutatorFlow_ module.

        Args:
            t_span (tuple): Time interval `(t0, t1)` for integrating the flow.
            step_size (float): Step size used during ODE integration.
            tau_par (Tensor or None): Optional initial value for the learnable
                tau parameter. If None, initializes to 1/4.
            mask (optional): A mask object that enables partial application
                of the flow by splitting the input and recombining it
                post-transformation.
            use_adjoint_method (bool): If True, uses the adjoint ODE solver;
                otherwise, uses the standard solver.
            **solver_kwargs: Additional arguments for the ODE solver.

        Note:
            The default settings are appropriate for `beta = 1` in the Wilson
            gauge action. For other values of `beta`, consider scaling
            the integration interval to `t_span = (0, beta)`.
        """
        super().__init__()

        # Choose between adjoint and standard ODE flow solver
        flow_class = AdjLieODEFlow_ if use_adjoint_method else LieODEFlow_

        # Initialize the ODE flow using ModalCommutatorDynamics
        self.flow_ = flow_class(
            ModalCommutatorDynamics(),
            t_span=t_span,
            step_size=step_size,
            **solver_kwargs
        )

        # Store mask object for handling partial flow
        self.mask = mask

        # Learnable tau parameter controlling flow strength
        if tau_par is None:
            self.tau_par = torch.nn.Parameter(0.25 * torch.ones(1))
        else:
            self.tau_par = tau_par

    def forward(self, eigvecs, *, eigangs, staples_object):
        """
        Applies the forward commutator-based flow to the input eigenvectors.

        Args:
            eigvecs (Tensor): Unitary eigenvector matrix; shape (..., N, N).
            eigangs (Tensor): Corresponding eigenvalue angles (phases),
                used to construct Lambda matrix; shape (..., N).
            staples_object: Object containing the SVD decomposition (Sigma
                matrix and and determinant angle).

        Returns:
            Tuple[Tensor, Tensor]: Transformed eigenvectors and log Jacobian.
        """
        # Extract the SVD determinant angle & unsqueeze it to match shape
        alpha = staples_object.svd_.rdet_angle.unsqueeze(-1)

        # Compute a modified eigenvalue vector using a cosine shift by alpha,
        # then enforce the zero-sum constraint
        lambda_vector = zero_sum_vector(torch.cos(eigangs + alpha))

        # Construct a Hermitian, traceless matrix from Sigma component of SVD
        sigma_matrix = hermitian_traceless(staples_object.svd_.Sigma)

        # Compute the flow time scale & group flow arguments into a tuple
        tau = self.tau_net()
        args = (lambda_vector, sigma_matrix, tau)

        # If no mask is defined, apply the flow transformation to whole data
        if self.mask is None:
            eigvecs, logj = self.flow_.forward(eigvecs, args=args)
        else:
            # Otherwise, split the data into masked and unmasked parts
            x_0, x_1 = self.mask.split(eigvecs)

            # Apply the flow only to the masked portion
            x_0, logj = self.flow_.forward(x_0, args=args)

            # Concatenate the transformed and untouched parts back together
            eigvecs = self.mask.cat(x_0, x_1)

        # Return the transformed eigenvectors and the log-determinant Jacobian
        return eigvecs, logj

    def reverse(self, eigvecs, *, eigangs, staples_object):
        """
        Applies the reverse commutator-based flow to the input eigenvectors.

        Args:
            eigvecs (Tensor): Unitary eigenvector matrix; shape (..., N, N).
            eigangs (Tensor): Corresponding eigenvalue angles (phases),
                used to construct Lambda matrix; shape (..., N).
            staples_object: Object containing the SVD decomposition (Sigma
                matrix and and determinant angle).

        Returns:
            Tuple[Tensor, Tensor]: Transformed eigenvectors and log Jacobian.
        """
        # Extract the SVD determinant angle & unsqueeze it to match shape
        alpha = staples_object.svd_.rdet_angle.unsqueeze(-1)

        # Compute a modified eigenvalue vector using a cosine shift by alpha,
        # then enforce the zero-sum constraint
        lambda_vector = zero_sum_vector(torch.cos(eigangs + alpha))

        # Construct a Hermitian, traceless matrix from Sigma component of SVD
        sigma_matrix = hermitian_traceless(staples_object.svd_.Sigma)

        # Compute the flow time scale & group flow arguments into a tuple
        tau = self.tau_net()
        args = (lambda_vector, sigma_matrix, tau)

        # If no mask is defined, apply the reversed flow to whole data
        if self.mask is None:
            eigvecs, logj = self.flow_.reverse(eigvecs, args=args)
        else:
            # Otherwise, split the data into masked and unmasked parts
            x_0, x_1 = self.mask.split(eigvecs)

            # Apply the reversed flow only to the masked portion
            x_0, logj = self.flow_.reverse(x_0, args=args)

            # Concatenate the transformed and untouched parts back together
            eigvecs = self.mask.cat(x_0, x_1)

        # Return the transformed eigenvectors and the log-determinant Jacobian
        return eigvecs, logj

    def tau_net(self):
        """
        Computes the flow-scaling parameter `tau`. For now it simply returns
        self.tau_par, but in general it can also depend on the frozen
        parameters such as lambda_vector and sigma_matrix.

        Returns:
            Tensor: tau used in flow dynamics.
        """
        return self.tau_par


# =============================================================================
class ModalCommutatorDynamics(AdjLieModule):
    r"""
    Implements the commutator-based flow dynamics (the ODE right-hand side) on
    the modal (eigenvector) matrix derived from the Wilson action in lattice
    gauge theory.

    This class extends `AdjLieModule` with analytic overrides for improved
    efficiency. See the module docstring for architectural details.

    The dynamics are governed by the following differential equation:

    .. math::

        \frac{dX}{dt} = - a [X \Lambda X^\dagger, \Sigma] X

    where:
        - `X` is the unitary matrix of eigenvectors (the modal matrix).
        - `Lambda` is a diagonal matrix of real eigenvalues.
        - `Sigma` is a Hermitian matrix.
        - `[A, B] = AB - BA` is the matrix commutator.
        - The overall coefficient is `a = tau / (2 n_c^2)`.
        - `tau` is a scalar controlling flow speed.
        - `n_c` is the number of colors of the gauge theory (dimension of `X`).

    Without loss of generality, the matrices `Lambda` and `Sigma` can be
    assumed traceless, as their traces do not affect the flow dynamics.
    However, when computing the log-Jacobian (see below), the trace components
    of `Lambda` and `Sigma` must be subtracted if they are nonzero.
    This subtraction introduces additional computational overhead.

    For efficiency in log-Jacobian computation and backpropagated gradients
    during training, this implementation assumes `Lambda` and `Sigma` are
    traceless by default. It is the user's responsibility to ensure this
    condition holds for all inputs.

    The above expression is equivalent to:

    .. math::

        \frac{dX}{dt} = - a (X \Lambda X^\dagger \Sigma - \text{h.c.}) X

    where h.c. denotes the Hermitian conjugate (i.e., conjugate transpose)
    of the preceding term. The expression inside the parentheses is manifestly
    anti-Hermitian.

    Both expressions are mathematically equivalent, but their behavior under
    automatic differentiation (AD) for computing complex-valued Jacobians can
    differ in subtle ways. In particular, the commutator form may lead to
    incorrect Jacobian estimates, such as an unintended factor of two, due to
    how PyTorch handles derivatives of complex-valued functions.
    In contrast, the anti-Hermitian form explicitly eliminates ambiguity and
    yields a reliable gradient computations under AD.

    This distinction is relevant only when AD is used for log-Jacobian
    calculations, as in `SlowModalCommutatorDynamics`, where users have
    the option to choose between the commutator form and the anti-Hermitian
    form for testing purposes.

    The log-Jacobian rate used in continuous normalizing flows is:

    .. math::

        \frac{d}{dt} \log |det J| = - 2 a (
            n_c Tr(X \Lambda X^\dagger \Sigma) - Tr(\Sigma) Tr(\Lambda)
        )

    The second term vanishes if `Lambda` and `Sigma` are traceless. As stated
    above, this implementation assumes traceless inputs to avoid unnecessary
    overhead in log-Jacobian and gradient computations.

    We stress again: it is the user's responsibility to ensure this condition
    holds for all inputs. Note that this requirement is handled internally by
    the `ModalCommutatorFlow_` class, which utilizes this class to implement
    the flow.

    Additional remarks:
        - `X`, `Lambda`, and `Sigma` are assumed to have a leading batch
          dimension, followed by additional axes corresponding to the lattice
          geometry.
        - Although `X` is ideally unitary, all formulas used for Jacobian
          computation and backpropagation remain valid even when `X` vanishes.
          This enables the use of this class in conjunction with masking over
          `X`. Therefore, when masking is applied, `X` may contain a mix of
          unitary and zero matrices.
    """

    return_logj_density = False

    def algebra_dynamics(self, t, x, lambda_vector, sigma_matrix, tau):
        r"""
        Computes the right-hand side of the ODE in the Lie algebra
        (tangent space) at time t.

        This method defines the flow in the Lie algebra corresponding:

        .. math::

            dX/dt X^\adjoint = - a (X \Lambda X^\dagger \Sigma - \text{h.c.})

        (See the class docstring for a full explanation.)

        Args:
            t (float): Time parameter (not used but included for API).
            x (Tensor): The modal matrix or zero matrix, or a combination.
            lambda_vector (Tensor): Real eigenvalue vector summing to zero.
            sigma_matrix (Tensor): Traceless Hermitian matrix.
            tau (float): Flow scaling factor.

        Here, `x`, `lambda_vector`, and `sigma_matrix` are assumed to have
        a leading batch dimension, followed by additional axes corresponding
        to the lattice geometry.

        Note:
            sigma_matrix and lambda_vector are supposed to be traceless and
            zero-summed, respectively. This assumption is used for computation
            efficiency, but it is the user's responsibility to ensure this
            assumption holds for inputs. While this assumption does not alter
            the flow itself, it does affect the computation of the log-Jacobian
            and the correctness of backpropagated gradients during training.

            It is the user's responsibility to ensure this conditions hold for
            all inputs.

        Returns:
            Tensor: The Lie algebra element (anti-Hermitian matrix)
            representing the instantaneous flow direction associated with x.
        """
        # Calculate the overall coefficient.
        n_c = x.shape[-1]
        coeff = - tau / (2 * n_c**2)
        # xlxs stands for X @ Lambda @ X^\dagger @ Sigma
        xlxs = inverse_eigh(lambda_vector, x) @ sigma_matrix
        return coeff * (xlxs - xlxs.adjoint())

    def calc_logj_rate(self, t, x, lambda_vector, sigma_matrix, tau):
        """
        Overrides the base class method to compute the log-Jacobian rate
        (i.e., the trace of the Jacobian df/dx) analytically.

        Unlike the original implementation, this version does NOT use
        the Hutchinson estimator to approximate the trace of the Jacobian
        df/dx but instead relies on a closed-form expression, improving
        efficiency and numerical stability.

        Args:
            t (float): Time parameter (not used but included for API).
            x (Tensor): The modal matrix or zero matrix, or a combination.
            lambda_vector (Tensor): Real eigenvalue vector summing to zero.
            sigma_matrix (Tensor): Traceless Hermitian matrix.
            tau (float): Flow scaling factor.

        Here, `x`, `lambda_vector`, and `sigma_matrix` are assumed to have
        a leading batch dimension, followed by additional axes corresponding
        to the lattice geometry.

        Note:
            sigma_matrix and lambda_vector are supposed to be traceless and
            zero-summed, respectively. This assumption is used for computation
            efficiency, but it is the user's responsibility to ensure this
            assumption holds for inputs. While this assumption does not alter
            the flow itself, it does affect the computation of the log-Jacobian
            and the correctness of backpropagated gradients during training.

        Returns:
            Tensor: Log-Jacobian rate of the flow.
        """
        # xlxs stands for X @ Lambda @ X^\dagger @ Sigma
        xlxs = inverse_eigh(lambda_vector, x) @ sigma_matrix
        trace = calc_trace(xlxs).real

        # Sum over all axes excluding the batch axis unless
        if not self.return_logj_density:
            trace = trace.reshape(trace.shape[0], -1).sum(dim=-1)

        n_c = x.shape[-1]
        coeff = - tau / (2 * n_c**2)

        # Total log-Jacobian rate formula
        return (2 * n_c * coeff) * trace

    def aug_reverse(self, t, aug_var, aug_frozen_var):
        """
        Overrides the base class method to compute reverse-time dynamics
        analytically for the augmented system in the adjoint method.

        Unlike the original implementation, this version does NOT use
        automatic differentiation but instead relies on a closed-form
        expression for the adjoint dynamics, improving efficiency and
        numerical stability.

        Args:
            t (float): Time parameter (not used but included for API).
            aug_var (TupleVar): Tuple of (state, adjoint of state) at time t.
            aug_frozen_var (TupleVar): Tuple containing:
                - frozen variables (lambda_vector, sigma_matrix, tau),
                - grad_logj: gradient of the loss w.r.t. log-Jacobian,
                - model parameters (unused here).

        Returns:
            TupleVar: A tuple of:
                - Time derivative of the (group-valued) state variable
                  (var_dot),
                - Time derivative of the (algebra-valued) adjoint variable
                  (grad_alg_var_dot).
        """
        # === Unpack ===
        x, grad_alg_var = aug_var.tuple
        (lambda_vector, sigma_matrix, tau), grad_logj = aug_frozen_var.tuple
        n_c = x.shape[-1]

        # === Compute algebra-valued and group-valued state dynamics ===
        coeff = - tau / (2 * n_c**2)

        xlx = inverse_eigh(lambda_vector, x)  # xlx: X @ Lambda @ X^\dagger
        xlxs = xlx @ sigma_matrix  # xlxs: X @ Lambda @ X^\dagger @ Sigma

        alg_var_dot = coeff * (xlxs - xlxs.adjoint())

        var_dot = alg_var_dot @ x

        # === Reshape grad_logj to broadcast correctly with other tensors ===
        grad_logj = grad_logj.view(-1, *(1,) * (x.ndim - 1))

        # === Compute algebra dynamics of the adjoint state analytically ===
        # Two terms contribute: to grad_alg_var_dot:
        # - From log-Jacobian term,
        # - From the adjoint-dynamcis contraction/tie with state dynamics
        from_contrac = coeff * commutator(xlx, sigma_matrix @ grad_alg_var)
        grad_alg_var_dot = (2 * n_c) * grad_logj * alg_var_dot + from_contrac

        # Return group-valued state & algebra-valued adjoint time derivatives
        return TupleVar(var_dot, grad_alg_var_dot)

    def calc_grad_params_rate(self, t, aug_var, aug_frozen_var):
        """
        Overrides the base class method to compute the rate of change of
        gradients with respect to parameters and frozen variables.

        Forms the Hamiltonian from the adjoint variables and system dynamics,
        then computes the gradient of the negative Hamiltonian with respect
        to model parameters.

        Unlike the original implementation, this version does NOT use
        automatic differentiation but instead relies on a closed-form
        expression for the adjoint dynamics, improving efficiency and
        numerical stability.

        Args:
            t (float): Time parameter (not used but included for API).
            aug_var (TupleVar): Tuple of (state, adjoint of state) at time t.
            aug_frozen_var (TupleVar): Tuple containing:
                - frozen variables (lambda_vector, sigma_matrix, tau),
                - grad_logj: gradient of the loss w.r.t. log-Jacobian,
                - model parameters (unused here).

        Returns:
            TupleVar: Gradients of the **negative** Hamiltonian with respect
            to the frozen variables, in the same order. Values are `None` for
            variables that do not require gradients.

        Note:
            The returned value represents the **negative** rate of change of
            the gradients with respect to the parameters. This sign convention
            arises because we are integrating the adjoint equations backward
            in time (from final time to initial time), which introduces a minus
            sign in the adjoint dynamics.
        """
        # === Unpack ===
        x, grad_alg = aug_var.tuple
        (lambda_vector, sigma_matrix, tau), grad_logj = aug_frozen_var.tuple
        n_c = x.shape[-1]

        # === Compute algebra dynamics ===
        coeff = -tau / (2 * n_c**2)
        xlx = inverse_eigh(lambda_vector, x)  # X @ Lambda @ X^\dagger

        # === Reshape grad_logj to broadcast correctly with other tensors ===
        grad_logj = grad_logj.view(-1, *(1,) * (x.ndim - 1))

        # === Compute & return negative gradients (reverse-time convention) ===
        common_args = (
            lambda_vector, sigma_matrix, tau, grad_alg, grad_logj, coeff, n_c
        )

        grad_params_rate = []

        if lambda_vector.requires_grad:
            grad_params_rate.append(self._grad_lambda(x, *common_args))

        if sigma_matrix.requires_grad:
            grad_params_rate.append(self._grad_sigma(xlx, *common_args))

        if tau.requires_grad:
            grad_params_rate.append(self._grad_tau(xlx, *common_args))

        return TupleVar(*grad_params_rate)

    # pylint: disable=unused-argument
    @staticmethod
    def _grad_lambda(x, lam, sigma, tau, grad_alg, grad_logj, coeff, n_c):
        """
        Computes **negative** of ∂H/∂lambda_vector analytically.
        """
        # The output is manifestly Hermitian; note grad_alg is anti-hermitian
        temp = n_c * grad_logj * sigma - sigma @ grad_alg
        return -coeff * (
            x.adjoint() @ (temp + temp.adjoint()) @ x
        ).diagonal(dim1=-2, dim2=-1).real

    # pylint: disable=unused-argument
    @staticmethod
    def _grad_sigma(xlx, lam, sigma, tau, grad_alg, grad_logj, coeff, n_c):
        """
        Computes **negative** of ∂H/∂sigma_matrix analytically.
        """
        # The output is manifestly Hermitian; note grad_alg is anti-hermitian
        temp = n_c * grad_logj * xlx - grad_alg @ xlx
        return -coeff * (temp + temp.adjoint())

    # pylint: disable=unused-argument
    @staticmethod
    def _grad_tau(xlx, lam, sigma, tau, grad_alg, grad_logj, coeff, n_c):
        """
        Computes **negative** of ∂H/∂tau analytically if tau.
        """
        xlxs = xlx @ sigma  # xlxs: X @ Lambda @ X^\dagger @ Sigma
        trace = calc_trace(xlxs).reshape(xlx.shape[0], -1).sum(dim=-1).real

        # Total Hamiltonian divided by tau, which is ∂H/∂tau
        coeff_prime = -1 / (2 * n_c**2)  # coeff / tau
        grad_tau = coeff_prime * (
            2 * n_c * torch.sum(grad_logj.ravel() * trace)
            +
            torch.sum(grad_alg.conj() * (xlxs - xlxs.adjoint())).real
        )

        return -grad_tau.reshape(*tau.shape)


# =============================================================================
class SlowModalCommutatorDynamics(AdjLieModule):
    r"""
    Reference implementation of modal commutator flow dynamics for testing and
    benchmarking against the optimized `ModalCommutatorDynamics`.

    This class defines only the flow dynamics (i.e., the right-hand side of
    the ODE in the Lie algebra) and inherits default implementations for
    log-Jacobian and adjoint gradient computations from the base class
    `AdjLieModule`, which uses PyTorch automatic differentiation.

    Intended for:
        - Validating the correctness of the closed-form version.
        - Benchmarking runtime and gradient accuracy.

    Args:
        use_commutator (bool):
            If True, computes the dynamics using the commutator form:

            .. math::

                \frac{dX}{dt} = - a [X \Lambda X^\dagger, \Sigma] X

            Otherwise, uses the expanded anti-Hermitian form:

            .. math::

               \frac{dX}{dt} = - a (X \Lambda X^\dagger \Sigma - \text{h.c.}) X

            These expressions are mathematically equivalent, but their behavior
            under automatic differentiation for calculation of complex Jacobian
            may differ in subtle ways.

            In particular, the commutator form may lead to incorrect Jacobian
            calculation (e.g., missing a factor of two) due to how PyTorch
            handles complex-valued derivatives of complex functions.
            The anti-Hermitian form ensures the right-hand side remains
            explicitly anti-Hermitian, avoiding ambiguity in gradient
            computation.

            This flag is intended purely for testing and internal verification.

    Note:
        This implementation is intentionally slower, as it relies on autograd
        rather than optimized closed-form expressions.
    """

    def __init__(self, use_commutator: bool = False):
        super().__init__(num_hutchinson_samples=None)
        self.use_commutator = use_commutator

    def algebra_dynamics(self, t, x, lambda_vector, sigma_matrix, tau):
        r"""
        Computes the right-hand side of the ODE in the Lie algebra
        (tangent space) at time t.

        (See `ModalCommutatorDynamics` for a full explanation.)
        """
        # Calculate the overall coefficient.
        n_c = x.shape[-1]
        coeff = - tau / (2 * n_c**2)

        # pylint: disable=no-else-return
        if self.use_commutator:
            # xlx stands for X @ Lambda @ X^\dagger
            xlx = inverse_eigh(lambda_vector, x)
            return coeff * commutator(xlx, sigma_matrix)
        else:
            # xlxs stands for X @ Lambda @ X^\dagger @ Sigma
            xlxs = inverse_eigh(lambda_vector, x) @ sigma_matrix
            return coeff * (xlxs - xlxs.adjoint())


# =============================================================================
def commutator(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """Returns the commutator of two square matrices `[A, B]`."""
    return mat1 @ mat2 - mat2 @ mat1


def hermitian_traceless(mtrx: torch.Tensor) -> torch.Tensor:
    """
    Project a square matrix (or batch of matrices) onto the Hermitian subspace.

    This function returns a Hermitian, traceless version of the input matrix.
    """
    # Make Hermitian
    mtrx = (mtrx + mtrx.adjoint()) / 2.

    # Compute average diagonal value (trace / n) over the last two axes
    reduced_trace = mtrx.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)

    # Subtract the average from the diagonal to make it traceless
    return mtrx - torch.diag_embed(reduced_trace.expand(mtrx.shape[:-1]))


def traceless_matrix(mtrx: torch.Tensor) -> torch.Tensor:
    """
    Project a square matrix or batch of matrices onto the subspace of traceless
    matrices.
    """
    # Compute average diagonal value (trace / n) over the last two axes
    reduced_trace = mtrx.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)

    # Subtract the average from the diagonal to make it traceless
    return mtrx - torch.diag_embed(reduced_trace.expand(mtrx.shape[:-1]))


def zero_sum_vector(vector: torch.Tensor) -> torch.Tensor:
    """
    Projects a vector or batch of vectors onto the subspace of zero-sum
    vectors, by subtracting the mean value of each vector from its elements.

    Args:
        vector (Tensor): Tensor of shape (..., n); n is the vector length.

    Returns:
        Tensor: Tensor of the same shape as input with zero-sum vectors.
    """
    return vector - vector.mean(dim=-1, keepdim=True)


def calc_trace(x: torch.Tensor) -> torch.Tensor:
    """Returns the trace of a square matrix."""
    return torch.sum(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)
