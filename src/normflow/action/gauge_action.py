# Copyright (c) 2021-2024 Javad Komijani

"""This is a module for defining gauge actions..."""


import torch

from numpy import pi

from ..lib.matrix_handles import WilsonStaplesHandle


# =============================================================================
class WilsonGaugeAction:
    r"""
    Wilson gauge action for SU(N_c) lattice gauge theory.

    The action is defined as:

    .. math::
        S = - \frac{\beta} {2 n_c} \sum_{\nu \neq \mu} Tr \text{Plaq}_{mu, nu}
          = - \frac{\beta} {n_c}   \sum_{\nu < \mu} ReTr \text{Plaq}_{mu, nu} .

    Here:
        - :math:`\beta` is the inverse coupling.
        - :math:`n_c` is the number of colors.
        - :math:`\text{Plaq}_{\mu\nu}` is the plaquette in the (μ, ν) plane.
    """

    def __init__(self, *, beta: float, ndim: int, n_c: int):
        """
        Initialize Wilson gauge action parameters.

        Args:
            beta (float): Gauge coupling parameter.
            ndim (int): Number of spacetime dimensions.
            n_c (int): Number of colors in the gauge group SU(n_c).
        """
        self.beta = beta
        self.ndim = ndim
        self.n_c = n_c
        self.staples_handle = WilsonStaplesHandle()

    def __call__(self, cfgs: torch.Tensor):
        """
        Evaluate the action for a batch of gauge configurations.

        Args:
            cfgs (torch.Tensor): Gauge link configurations of shape
            [batch, ndim, ...].

        Returns:
            torch.Tensor: Scalar or per-batch action values.
        """
        return self.action(cfgs)

    def action(self, cfgs: torch.Tensor):
        """
        Compute the Wilson gauge action for given configurations.

        Args:
            cfgs (torch.Tensor): Gauge link configurations.

        Returns:
            torch.Tensor: Action value(s) for the input configurations.
        """
        dim = tuple(range(1, 1 + self.ndim))  # sum over spatial dimensions
        sum_ = 0
        for mu in range(1, self.ndim):
            for nu in range(mu):
                sum_ += torch.sum(self.calc_plaq(cfgs, mu=mu, nu=nu), dim=dim)
        # Note: 1/n_c factor is already included in `calc_plaq`.
        return -self.beta * sum_

    def force(self, links: torch.Tensor):
        """
        Compute the gauge force: minus gradient of action w.r.t. gauge link
        variables.

        Args:
            links (torch.Tensor): Gauge link variables.

        Returns:
            torch.Tensor: Force on each link.
        """
        # The algebra force is multiplied by links to map back to group space
        return self.algebra_force(links) @ links

    def algebra_force(self, links: torch.Tensor):
        """
        Compute the minus gradient of the action w.r.t. algebra-valued gauge
        variables.

        Args:
            links (torch.Tensor): Gauge link variables.

        Returns:
            torch.Tensor: Anti-Hermitian traceless force matrices in the Lie
            algebra.

        Note:
            The magnitude of this force depends on the normalization of
            the SU(N_c) generators T^a. Lattice QCD literature often uses
            Tr(T^a T^b) = -1/2 δ^ab, but this code uses Tr(T^a T^b) = -δ^ab.
        """
        ndim = self.ndim
        stps = torch.zeros_like(links)  # Sum of staples for each link
        for mu in range(ndim):
            kws = {'mu': mu, 'nu_list': [nu for nu in range(ndim) if nu != mu]}
            stps[:, mu] = self.staples_handle.calc_staples_sum(links, **kws)

        coeff = -self.beta / self.n_c
        algebra_force = coeff * anti_hermitian_traceless(links @ stps)
        return algebra_force

    def calc_plaq(self, cfgs: torch.Tensor, *, mu: int, nu: int, real=True):
        """
        Compute the reduced trace of plaquettes in a given μ-ν plane.

        Args:
            cfgs (torch.Tensor): Gauge configurations.
            mu (int): First spacetime direction.
            nu (int): Second spacetime direction.
            real (bool, optional): Return only the real part. Default is True.

        Returns:
            torch.Tensor: Reduced trace of plaquettes.
        """
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
        """
        Multiply links around a plaquette and compute the reduced trace.

        Args:
            a, b, c, d (torch.Tensor): Link matrices forming the plaquette.

        Returns:
            torch.Tensor: Reduced trace of the plaquette product.
        """
        mul = torch.matmul
        plaq = mul(mul(a, b), mul(d, c).adjoint())
        return calc_reduced_trace(plaq)

    def log_prob(self, x, action_logz=0):
        """Returns log probability up to an additive constant."""
        return -self.action(x) - action_logz

    @property
    def parameters(self):
        """Return the parameters of the action."""
        return {'beta': self.beta, 'ndim': self.ndim}


# =============================================================================
class U1WilsonGaugeAction(WilsonGaugeAction):
    """A special case of GaugeAction with special `plaq_rule`, ...."""

    def __init__(self, **kwargs):
        super().__init__(n_c=1, **kwargs)

    @staticmethod
    def plaq_rule(a, b, c, d):
        """Mutiply the links and calculate the reduced trace of the product."""
        return a * b * torch.conj(d * c)

    def calc_topo_charge(self, cfgs):
        """Compute the topological charge."""
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
    """Compute the trace of the input matrix x."""
    return torch.sum(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)


def calc_reduced_trace(x):  # reduced trace = 1/n trace()
    """Compute the reduced trace of the input matrix x."""
    return torch.mean(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)


def anti_hermitian_traceless(mtrx):
    """Project the the input matrix to the space of anti-Hermitian matrices."""
    mtrx = (mtrx - mtrx.adjoint()) / 2.
    mu = torch.mean(torch.linalg.diagonal(mtrx), dim=-1, keepdim=True)
    mu = torch.diag_embed(torch.repeat_interleave(mu, mtrx.shape[-1], dim=-1))
    return mtrx - mu
