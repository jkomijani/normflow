# Copyright (c) 2021-2024 Javad Komijani

"""This is a module for defining actions for scalar theories."""


import torch

Tensor = torch.Tensor


class ScalarPhi4Action:
    r"""
    A class representing the scalar $\phi^4$ field theory action.

    The action is defined as:

    .. math::

        S = \sum_{x} \left(
            \frac{\kappa}{2} \sum_\mu (\Delta_\mu \phi(x))^2
            + \frac{m^2}{2} \phi(x)^2
            + \lambda \phi(x)^4
        \right)

    This class allows computation of the action, action density, and related
    quantities for a given set of field variables using parameters for the mass
    term, quartic coupling, and nearest-neighbor interactions.

    Parameters
    ----------
    m_sq : float
        The mass-squared parameter, $m^2$, which controls the quadratic term in
        the action.

    lambd : float
        The quartic coupling parameter, $\lambda$, which controls the $\phi^4$
        interaction.

    kappa : float, optional
        The nearest-neighbor interaction parameter, $\kappa$. Default is 1.0.

    Methods
    -------
    __call__(var: Tensor) -> Tensor
        Computes the action for a given input tensor of field variables by
        calling the `action` method.

    action(var: Tensor) -> Tensor
        Computes the action for a given input tensor of field variables.
        The first axis in `var` is assumed to be the batch axis.

    action_density(var: Tensor) -> Tensor
        Computes the action density with respect to the input field variables.
        The returned action density is symmetric and has a positive kinetic term.

    score(var: Tensor) -> Tensor
        Computes the gradient of the log-likelihood, which is the negative of
        the action, evaluated with respect to the input field variables.

    potential(var: Tensor) -> Tensor
        Computes the potential energy for the input field variables based on
        the mass-squared and quartic coupling parameters.

    log_prob(var: Tensor, action_logz=0) -> Tensor
        Computes the log probability of corresponding PDF up to an additive
        constant, calculated as the negative of the action.
    """

    def __init__(self, *, m_sq: float, lambd: float, kappa: float = 1.0):

        if kappa is None:
            kappa = 0

        self.kappa = kappa
        self.m_sq = m_sq
        self.lambd = lambd

    def __call__(self, var: Tensor):
        return self.action(var)

    def action(self, var: Tensor):
        """Computes the action for the given input tensor of field variables.

        The first axis in `var` is supposed to be the batch axis.
        """
        ndim = var.ndim - 1  # 0 axis -> batch axis

        w_0 = self.kappa
        w_2 = 0.5 * self.m_sq + self.kappa * ndim
        w_4 = self.lambd

        dim = tuple(range(1, 1 + ndim))
        action = torch.sum(w_2 * var**2 + w_4 * var**4, dim=dim)

        for mu in dim:
            if w_0 == 0:
                break
            action -= w_0 * torch.sum(var * torch.roll(var, 1, mu), dim=dim)

        return action

    def action_density(self, var: Tensor):
        """Computes the action density for the given input tensor of field
        variables.

        Note that the action density is not unique; a version is returned that
        is symmetric and also its kinetic term is always positive.

        The first axis in `var` is supposed to be the batch axis.
        """
        ndim = var.ndim - 1  # 0 axis -> batch axis

        w_0 = self.kappa
        w_2d = 0.5 * self.m_sq  # different from w_2 in action
        w_4 = self.lambd

        action_density = w_2d * var**2 + w_4 * var**4

        for mu in dim:
            if w_0 == 0:
                break
            action_density += (w_0 / 4) * (var - torch.roll(var, -1, mu))**2
            action_density += (w_0 / 4) * (var - torch.roll(var, +1, mu))**2

        return action_density

    def score(self, var: Tensor):
        """Computes the gradient of the log-likelihood, which is the negative
        of the action, evaluated with respect to the input field variables.
        """
        ndim = var.ndim - 1  # 0 axis -> batch axis

        w_0 = self.kappa
        w_2 = 0.5 * self.m_sq + self.kappa * ndim
        w_4 = self.lambd

        score = -(2 * w_2) * var**2 - (4 * w_4) * var**3

        for mu in dim:
            if w_0 == 0:
                break
            score = w_0 * torch.roll(var, -1, mu)
            score = w_0 * torch.roll(var, +1, mu)

        return score

    def potential(self, var: Tensor):
        """Computes the potential energy for the input field variables based on
        the mass-squared and quartic coupling parameters.
        """
        return self.m_sq / 2 * var**2 + self.lambd * var**4

    def log_prob(self, var: Tensor, action_logz=0):
        """Computes the log probability up to an additive constant, calculated
        as the negative of the action.
        """
        return -self.action(var) - action_logz
