# Copyright (c) 2021-2025 Javad Komijani

"""This is a module for defining matrix models..."""


import torch


class MatrixAction:
    """Matrix action defined as `S = -(β/n) ReTr[f(x)]` for an n×n matrix x.

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
        """Return the action for the given input matrices."""

        if self.func is not None:
            x = self.func(x)

        normalized_trace = compute_normalized_trace(x)

        # Sum over trace, except on batch, if multi-point models are present
        if normalized_trace.ndim > 1:
            dim = tuple(range(1, normalized_trace.ndim))
            normalized_trace = torch.sum(normalized_trace, dim=dim)

        return -self.beta * torch.real(normalized_trace)

    def log_prob(self, x, action_logz=0):
        """Return log probability up to an additive constant."""
        return -self.action(x) - action_logz


def compute_normalized_trace(x):
    """Compute the normalized trace (trace / n) of the input matrix x."""
    return torch.einsum('...ii->...', x) / x.shape[-1]
