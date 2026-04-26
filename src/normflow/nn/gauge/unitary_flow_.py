# Copyright (c) 2024 Javad Komijani

"""This module contains new neural networks for transforming matrices.

The classes defined here are subclasses of Module_, and like Module_, the
trailing underscore implies that the associated forward and reverse methods
handle the Jacobians of the transformation.
"""

# pylint: disable=invalid-name, relative-beyond-top-level

import torch
import numpy as np

from .modal_commutator_odeflow_ import hermitian_traceless

from .._core import Module_
from ...lib.matrix_handles import UnitaryFlow_
from ...lib.matrix_handles import transform_modal2antihermitian2unitary


__all__ = ["ModalMatrixSteppedCommutatorFlow_"]


class ModalMatrixSteppedCommutatorFlow_(Module_):
    r"""
    Stepped version of a commutator-based modal flow.

    This is a discrete approximation of the continuous ODE acting on a
    modal matrix X (eigenvectors):

    .. math::

        \frac{dX}{dt} = -a [X \Lambda X^\dagger, \Sigma] X

    where:
        - X is unitary (possibly masked),
        - \Lambda is a traceless diagonal matrix of real eigenvalues,
        - \Sigma is a traceless Hermitian matrix,
        - [A, B] = AB - BA is the commutator,
        - a is a scalar scale factor.

    Discrete flow
    -------------
    The implementation replaces the ODE with n_steps unitary updates:

        X ← F_k(X) ... F_2(X) F_1(X) X

    Each step is unitary and approximates the continuous generator.

    Inverse map
    -----------
    The inverse is not available in closed form and is computed only
    iteratively via fixed-point refinement of the forward step.
    """

    # Single-step unitary update constructed from an anti-Hermitian generator.
    # reverse_mode_iter controls the number of fixed-point iterations used
    # to approximate the inverse map.
    flow_ = UnitaryFlow_(
        func=transform_modal2antihermitian2unitary,
        n_steps=1,
        reverse_mode_iter=10
    )

    def __init__(self, tau_net=None, tau_par=torch.zeros(1), mask=None):
        super().__init__()
        self.mask = mask
        if tau_net is None:
            self.tau_par = tau_par
        else:
            self.tau_net = tau_net

    def forward(self, state, staples_ctx):
        """
        Apply the stepped commutator flow on state.eigvecs.

        Args:
            state (SpectralState): Contains eigangs, eigvecs, and logj.
            staples_ctx (object): Staple context (cached decompositions).

        Returns:
            SpectralState: Updated spectral state after transformation.
        """
        # Construct Λ (real diagonal encoded as a vector, minus its mean)
        diag_Lambda = torch.cos(state.eigangs) + 0j

        # Σ is a Hermitian generator coming from the SVD context
        Sigma = hermitian_traceless(staples_ctx.svd_result.sigma_matrix_factor)

        # Step size τ (learned or fixed)
        tau = self.tau_net(phase=state.eigangs)

        kwargs = {'diag_Lambda': diag_Lambda, 'Sigma': Sigma, 'tau': tau}

        if self.mask is None:
            eigvecs, logJ_density = self.flow_.forward(state.eigvecs, **kwargs)
        else:
            # Apply flow only to a sub-block, then reconstruct full matrix
            x_0, x_1 = self.mask.split(state.eigvecs)
            x_0, logJ_density = self.flow_.forward(x_0, **kwargs)

            # Enforce mask constraints after update
            x_0 = self.mask.purify(x_0, channel=0)
            eigvecs = self.mask.cat(x_0, x_1)

            # Enforce mask constraints to log-Jacobian too
            logJ_density = logJ_density.reshape(*logJ_density.shape, 1, 1)
            logJ_density = self.mask.purify(logJ_density, channel=0)

        logJ = self.sum_density(logJ_density)

        state.eigvecs = eigvecs
        state.logj += logJ

        return state

    def reverse(self, state, staples_ctx):
        """
        Approximate inverse of the stepped commutator flow.

        Uses fixed-point iterations to invert each unitary step.

        Args:
            state (SpectralState): Contains eigangs, eigvecs, and logj.
            staples_ctx (object): Staple context (cached decompositions).

        Returns:
            SpectralState: Updated spectral state after inverse transformation.
        """
        # Construct Λ (real diagonal encoded as a vector, minus its mean)
        diag_Lambda = torch.cos(state.eigangs) + 0j

        # Σ is a Hermitian generator coming from the SVD context
        Sigma = hermitian_traceless(staples_ctx.svd_result.sigma_matrix_factor)

        # Step size τ (learned or fixed)
        tau = self.tau_net(phase=state.eigangs)

        kwargs = {'diag_Lambda': diag_Lambda, 'Sigma': Sigma, 'tau': tau}

        if self.mask is None:
            eigvecs, logJ_density = self.flow_.reverse(state.eigvecs, **kwargs)
        else:
            # Apply flow only to a sub-block, then reconstruct full matrix
            x_0, x_1 = self.mask.split(state.eigvecs)
            x_0, logJ_density = self.flow_.reverse(x_0, **kwargs)

            # Enforce mask constraints after update
            x_0 = self.mask.purify(x_0, channel=0)
            eigvecs = self.mask.cat(x_0, x_1)

            # Enforce mask constraints to log-Jacobian too
            logJ_density = logJ_density.reshape(*logJ_density.shape, 1, 1)
            logJ_density = self.mask.purify(logJ_density, channel=0)

        logJ = self.sum_density(logJ_density)

        state.eigvecs = eigvecs
        state.logj += logJ

        return state

    def tau_net(self, **dummy_kwargs):
        """
        Default step size.

        Returns a positive scalar (via softplus) scaled by (1 / 4π)^2.
        """
        # Note that `transform_modal2antihermitian2unitary` has a negative sign
        # computing `exp(-τ [H, Σ])`, indicating τ is positive and the flow
        # matches the sign convention in the continuous commutator ODE.
        return torch.nn.functional.softplus(self.tau_par) / (4 * np.pi)**2


class SUnSpectralFlow_(Module_):
    # OBSOLETE: see the version before 18/Apr/2026
    """NOT RELIABLE YET; see the `integrate` method."""

    def __init__(self, tau_net=None, tau_par=torch.zeros(1), mask=None):
        super().__init__()
        self.mask = mask
        if tau_net is None:
            self.tau_par = tau_par
        else:
            self.tau_net = tau_net

    def forward(self, eigangs, *, eigvecs, staples_object, positive_tau=True):
        sigma = torch.linalg.diagonal(
                eigvecs.adjoint() @ staples_object.svd_.Sigma @ eigvecs
                ).real
        alpha = staples_object.svd_.rdet_angle.unsqueeze(-1)
        if positive_tau:
            tau = self.tau_net(phase=(eigangs + alpha))
        else:
            tau = - self.tau_net(phase=(eigangs + alpha))
        kwargs = dict(alpha=alpha, s=sigma, t=tau)
        if self.mask is None:
            eigangs, logJ_density = self.integrate(eigangs, **kwargs)
        else:
            mask = self.mask
            x_0, x_1 = mask.split(eigangs)
            x_0, logJ_density = self.integrate(x_0, **kwargs)
            x_0 = mask.purify(x_0, channel=0)
            eigangs = mask.cat(x_0, x_1)
            logJ_density = mask.purify(logJ_density.unsqueeze(-1), channel=0)
        logJ = self.sum_density(logJ_density)
        return eigangs, logJ

    def reverse(self, *args, **kwargs):
        return self.forward(*args, **kwargs, positive_tau=False)

    def tau_net(self, **kwargs):  # this is just the default self.tau_net
        return 0.02 * torch.nn.functional.softplus(self.tau_par)

    def integrate(self, x, *, s, t, alpha):
        r"""Integrate :math:`\frac{dx}{dt} = - s \sin(x)` with the condition
        that the sum of :math:`x` (over the last axis) remains constant.

        The prescription used here is to eliminate one of the elements of `x`
        in favor of the others. The inverse transformation is known in closed
        form if the ordering of eigenvalues does not change. Otherwise, the
        transformation is not invertible.

        Here is another scheme:
        1.  flow the difference of the first and second elements,
        2.  flow all other elements,
        3.  construct first and second elements from the difference, obtained
            in step 1, and the sum of other elements, obtained in step 2.
        """
        x += alpha
        mu = torch.sum(x, dim=-1, keepdim=True)  # mu = const is the constraint
        x, grad = self.exact_decoupled_solution(
            x[..., :-1], s = s[..., :-1], t = t[..., :-1] if t.ndim > 1 else t
            )
        x = torch.cat([x, mu - torch.sum(x, dim=-1, keepdim=True)], dim=-1)
        x -= alpha
        return x, torch.sum(torch.log(grad), dim=-1)

    @staticmethod
    def exact_decoupled_solution(x, *, s, t):
        r"""Return the exact solution of :math:`\frac{dx}{dt} = - s \sin(x)`,
        which is the asymptotic flow equation as $s$ tends to infinity,
        integrated from zero to :math:`tau`. The closed-form solution is

        .. math::

            \tan[x(t) / 2] = e^{-s t} \tan[x(0) / 2] .
        """
        tanxf = torch.tan(x / 2)  # f for half!
        coef = torch.exp(-s * t)
        x = 2 * torch.atan(coef * tanxf)
        grad = coef * (1 + tanxf**2) / (1 + (coef * tanxf)**2)  # dx(t) / dx(0)

        return x, grad

        # shift_term = torch.zeros_like(x)
        # shift_term[x > np.pi] = 2 * np.pi
        # shift_term[x < -np.pi] = - 2 * np.pi

        # the following lines are not compatible with automatic differentiation
        # x[x_0 == np.pi] = np.pi
        # x[x_0 == -np.pi] = -np.pi

        # return x + shift_term, grad


class SU2SpectralFlow_(SUnSpectralFlow_):
    """Unlike the superclass, this class is reliable."""

    def integrate(self, x, *, s, t, alpha):
        r"""Return the solution of :math:`\frac{dx}{dt} = - s \sin(x)`"""
        x, grad = self.exact_decoupled_solution(x, s=s, t=t)
        return x, torch.log(grad[..., 0])
