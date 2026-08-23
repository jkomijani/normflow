# Copyright (c) 2026 Javad Komijani

"""This is a module for defining U(1) models..."""


import torch


class PhasorAction:
    """
    Phasor action defined as `S = β Re[1 - f(X)]` for a U(1) variable X.

    Unlike MatrixAction, X is a plain complex phasor with no matrix structure.

    Args:
        beta (float): Coupling constant `β` in the action.
        func (callable, optional): Function `f` applied to X; identity if None.
    """

    def __init__(self, beta, func=None):
        self.beta = beta
        self.func = func

    def __call__(self, x):
        """Evaluate and return action."""
        return self.action(x)

    def action(self, x):
        """Return the action for the given input phasor variables."""

        if self.func is not None:
            x = self.func(x)

        action_density = 1 - x.real

        # Sum over all but the batch axis, if multi-point models are present
        if action_density.ndim > 1:
            dim = tuple(range(1, action_density.ndim))
            action_density = torch.sum(action_density, dim=dim)

        return self.beta * action_density

    def log_prob(self, x, action_logz=0):
        """Return log probability up to an additive constant."""
        return -self.action(x) - action_logz
