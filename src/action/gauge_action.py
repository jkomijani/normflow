# Copyright (c) 2021-2024 Javad Komijani

"""This is a module for defining gauge actions..."""


import torch

from numpy import pi

from ..lib.matrix_handles import WilsonStaplesHandle


# =============================================================================
class WilsonGaugeAction:
    r"""The Wilson gauge action is defined as

    .. math::

        S = - \frac{\beta} {2 n_c} \sum_{\nu \neq \mu} Tr \text{Plaq}_{mu, nu}
          = - \frac{\beta} {n_c}   \sum_{\nu < \mu} ReTr \text{Plaq}_{mu, nu} .
    """
    def __init__(self, *, beta, ndim, n_c):
        self.beta = beta
        self.ndim = ndim
        self.n_c = n_c  # number of colors of the gauge
        self.staples_handle = WilsonStaplesHandle()

    def reset_parameters(self, *, beta):
        self.beta = beta
       
    def __call__(self, cfgs):
        return self.action(cfgs)
  
    def action(self, cfgs):
        """Returns action corresponding to input configurations."""
        dim = tuple(range(1, 1 + self.ndim))  # 0 axis is the batch axis
        sum_ = 0
        for mu in range(1, self.ndim):
            for nu in range(mu):
                sum_ += torch.sum(self.calc_plaq(cfgs, mu=mu, nu=nu), dim=dim)
        # 1 / n_c is considered in 'self.calc_plaq`; see `calc_reduced_trace`.
        return - self.beta * sum_

    def action_density(self, cfgs):
        """Returns action density corresponding to input configurations."""
        dim = tuple(range(1, 1 + self.ndim))  # 0 axis is the batch axis
        density = 0
        for mu in range(1, self.ndim):
            for nu in range(mu):
                density += self.calc_plaq(cfgs, mu=mu, nu=nu)
        # 1 / n_c is considered in 'self.calc_plaq`; see `calc_reduced_trace`.
        return - self.beta * action_density

    def force(self, links):
        """Force is minus derivative of action w.r.t. gauge variables."""
        return self.algebra_force(links) @ links

    def algebra_force(self, links):
        """Force is minus derivative of action w.r.t. (algebra) gauge variables
        """
        ndim = self.ndim
        stps = torch.zeros_like(links)  # staples sum
        for mu in range(ndim):
            dict_ = dict(mu=mu, nu_list=[nu for nu in range(ndim) if nu != mu])
            stps[:, mu] = self.staples_handle.calc_staples_sum(links, **dict_)
        coeff = - self.beta / (2 * self.n_c)
        algebra_force = coeff * anti_hermitian_traceless(links @ stps)
        return algebra_force

    def calc_plaq(self, cfgs, *, mu, nu, real=True):
        x_mu = cfgs[:, mu]
        x_nu = cfgs[:, nu]
        plaq = self.plaq_rule(
                x_mu,
                torch.roll(x_nu, -1, dims=1 + mu),
                torch.roll(x_mu, -1, dims=1 + nu),
                x_nu
                )
        return torch.real(plaq) if real else plaq

    @staticmethod
    def plaq_rule(a, b, c, d):
        mul = torch.matmul
        plaq = mul(mul(a, b), mul(d, c).adjoint())
        return calc_reduced_trace(plaq)

    def log_prob(self, x, action_logz=0):
        """Returns log probability up to an additive constant."""
        return -self.action(x) - action_logz

    @property
    def parameters(self):
        return dict(beta=self.beta, ndim=self.ndim)


# =============================================================================
class U1WilsonGaugeAction(WilsonGaugeAction):
    """A special case of GaugeAction with special `plaq_rule`, ...."""

    def __init__(self, **kwargs):
        super().__init__(n_c=1, **kwargs)

    @staticmethod
    def plaq_rule(a, b, c, d):
        return a * b * torch.conj(d * c)

    def calc_topo_charge(self, cfgs):
        topo_charge = 0
        for mu in range(1, self.ndim):
            for nu in range(mu):
                angle_plaq = torch.angle(
                                self.calc_plaq(cfgs, mu=mu, nu=nu, real=False)
                                )
                dim = tuple(range(1, len(angle_plaq.shape)))
                topo_charge += torch.sum(angle_plaq, dim=dim) / (2 * pi)
        return topo_charge


# =============================================================================
def calc_trace(x):
    return torch.sum(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)


def calc_reduced_trace(x):  # reduced trace = 1/n trace()
    return torch.mean(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)


def anti_hermitian_traceless(mtrx):
    mtrx = (mtrx - mtrx.adjoint()) / 2.
    mu = torch.mean(torch.linalg.diagonal(mtrx), dim=-1, keepdim=True)
    mu = torch.diag_embed(torch.repeat_interleave(mu, mtrx.shape[-1], dim=-1))
    return mtrx - mu
