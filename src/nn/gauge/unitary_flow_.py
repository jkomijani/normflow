# Copyright (c) 2024 Javad Komijani

"""This module contains new neural networks for transforming matrices.

The classes defined here are subclasses of Module_, and like Module_, the
trailing underscore implies that the associated forward and backward methods
handle the Jacobians of the transformation.
"""

import torch
import numpy as np

from .._core import Module_
from ...lib.matrix_handles import UnitaryFlow_
from ...lib.matrix_handles import flow_handle


class ModalMatrixFlow_(Module_):

    flow_ = UnitaryFlow_(
            func = flow_handle.modal2antihermitian2unitary,
            reverse_mode_iter = 10
            )

    def __init__(self, tau_net=None, tau_par=torch.zeros(1), mask=None):
        super().__init__()
        self.mask = mask
        if tau_net is None:
            self.tau_par = tau_par
        else:
            self.tau_net = tau_net

    def forward(self, eigvecs, *, eigangs, staples_object):
        alpha = staples_object.svd_.rdet_angle.unsqueeze(-1)
        lambda_ = torch.cos(eigangs + alpha) + 0j
        sigma = staples_object.svd_.Sigma
        kwargs = dict(lambda_=lambda_, sigma=sigma)
        kwargs['tau'] = self.tau_net(phase=(eigangs + alpha))
        if self.mask is None:
            eigvecs, logJ_density = self.flow_(eigvecs, **kwargs)
        else:
            x_0, x_1 = self.mask.split(eigvecs)
            x_0, logJ_density = self.flow_(x_0, **kwargs)
            x_0 = self.mask.purify(x_0, channel=0)
            eigvecs = self.mask.cat(x_0, x_1)
            logJ_density = logJ_density.reshape(*logJ_density.shape, 1, 1)
            logJ_density = self.mask.purify(logJ_density, channel=0)
        logJ = self.sum_density(logJ_density)
        return eigvecs, logJ

    def reverse(self, eigvecs, *, eigangs, staples_object):
        alpha = staples_object.svd_.rdet_angle.unsqueeze(-1)
        lambda_ = torch.cos(eigangs + alpha) + 0j
        sigma = staples_object.svd_.Sigma
        kwargs = dict(lambda_=lambda_, sigma=sigma)
        kwargs['tau'] = self.tau_net(phase=(eigangs + alpha))
        if self.mask is None:
            eigvecs, logJ_density = self.flow_.reverse(eigvecs, **kwargs)
        else:
            x_0, x_1 = self.mask.split(eigvecs)
            x_0, logJ_density = self.flow_.reverse(x_0, **kwargs)
            x_0 = self.mask.purify(x_0, channel=0)
            eigvecs = self.mask.cat(x_0, x_1)
            logJ_density = logJ_density.reshape(*logJ_density.shape, 1, 1)
            logJ_density = self.mask.purify(logJ_density, channel=0)
        logJ = self.sum_density(logJ_density)
        return eigvecs, logJ

    def tau_net(self, **kwargs):  # this is just the default self.tau_net
        return 0.02 * torch.nn.functional.softplus(self.tau_par)


class SUnSpectralFlow_(Module_):
    """NOT RELIABLE YET except for SU(2); see the `solve` method."""

    def __init__(self, tau_net=None, tau_par=torch.zeros(1), mask=None):
        super().__init__()
        self.mask = mask
        if tau_net is None:
            self.tau_par = tau_par
        else:
            self.tau_net = tau_net

    def forward(self, eigangs, *, eigvecs, staples_object):
        sigma = torch.linalg.diagonal(
                eigvecs.adjoint() @ staples_object.svd_.Sigma @ eigvecs
                ).real
        alpha = staples_object.svd_.rdet_angle.unsqueeze(-1)
        tau = self.tau_net(phase=(eigangs + alpha))
        kwargs = dict(alpha=alpha, s=sigma, t=tau)
        if self.mask is None:
            eigangs, logJ_density = self.solve(eigangs, **kwargs)
        else:
            mask = self.mask
            x_0, x_1 = mask.split(eigangs)
            x_0, logJ_density = self.solve(x_0, **kwargs)
            x_0 = mask.purify(x_0, channel=0)
            eigangs = mask.cat(x_0, x_1)
            logJ_density = mask.purify(logJ_density.unsqueeze(-1), channel=0)
        logJ = self.sum_density(logJ_density)
        return eigangs, logJ

    def reverse(self, eigangs, *, eigvecs, staples_object):
        sigma = torch.linalg.diagonal(
                eigvecs.adjoint() @ staples_object.svd_.Sigma @ eigvecs
                ).real
        alpha = staples_object.svd_.rdet_angle.unsqueeze(-1)
        tau = self.tau_net(phase=(eigangs + alpha))
        kwargs = dict(alpha=alpha, s=sigma, t=-tau)
        if self.mask is None:
            eigangs, logJ_density = self.solve(eigangs, **kwargs)
        else:
            mask = self.mask
            x_0, x_1 = mask.split(eigangs)
            x_0, logJ_density = self.solve(x_0, **kwargs)
            x_0 = mask.purify(x_0, channel=0)
            eigangs = mask.cat(x_0, x_1)
            logJ_density = mask.purify(logJ_density.unsqueeze(-1), channel=0)
        logJ = self.sum_density(logJ_density)
        return eigangs, logJ

    def tau_net(self, **kwargs):  # this is just the default self.tau_net
        return 0.02 * torch.nn.functional.softplus(self.tau_par)

    def solve(self, x, *, s, t, alpha):
        r"""Return the solution of :math:`\frac{dx}{dt} = - s \sin(x)` with the
        condition that the sum of x (over the last axis) remains constant.

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
        x = torch.cat([x, mu - torch.sum(x, dim=-1, keepdim=True)], dim =-1)
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
        shift_term = torch.zeros_like(x)
        shift_term[x > np.pi] = 2 * np.pi
        shift_term[x < -np.pi] = - 2 * np.pi

        tanxf = torch.tan(x / 2)  # f for half!
        coef = torch.exp(-s * t)
        x = 2 * torch.atan(coef * tanxf)
        grad = coef * (1 + tanxf**2) / (1 + (coef * tanxf)**2)  # dx(t) / dx(0)

        # the following lines are not compatible with automatic differentiation
        # x[x_0 == np.pi] = np.pi
        # x[x_0 == -np.pi] = -np.pi

        return x + shift_term, grad


class SU2SpectralFlow_(SUnSpectralFlow_):

    def solve(self, x, *, s, t, alpha):
        r"""Return the solution of :math:`\frac{dx}{dt} = - s \sin(x)`"""
        x, grad = self.exact_decoupled_solution(x, s=s, t=t)
        return x, torch.log(grad[..., 0])
