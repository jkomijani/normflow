# Copyright (c) 2026 Javad Komijani

"""This is a module for defining U(1) models..."""


import torch


class U1Action:
    """
    U(1) action defined as `S = -β Re[f(x)]` for a U(1) variable x.

    Unlike `MatrixAction`, x is a plain complex phase with no matrix structure.

    Args:
        beta (float): Coupling constant `β` in the action.
        func (callable, optional): Function `f` applied to x.
    """

    def __init__(self, beta, func=None):
        self.beta = beta
        self.func = func

    def __call__(self, x):
        """Evaluate and return action."""
        return self.action(x)

    def action(self, x):
        """Return the action for the given input U(1) variables."""

        if self.func is not None:
            x = self.func(x)

        # Sum over all but the batch axis, if multi-point models are present
        if x.ndim > 1:
            dim = tuple(range(1, x.ndim))
            x = torch.sum(x, dim=dim)

        return -self.beta * torch.real(x)

    def log_prob(self, x, action_logz=0):
        """Return log probability up to an additive constant."""
        return -self.action(x) - action_logz
