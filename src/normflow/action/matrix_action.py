# Copyright (c) 2021-2025 Javad Komijani

"""This is a module for defining matrix models..."""


import torch


class MatrixAction:
    """Matrix action defined as `S = -(β/n) ReTr[f(x g)]` for an n×n matrix x.

    Args:
        beta (float): Coupling constant `β` in the action.
        staples_matrix (torch.Tensor): Constant matrix `g`.
        func (callable, optional): Function `f` applied to the matrix product.
    """

    def __init__(self, beta, staples_matrix=None, func=None):
        self.beta = beta
        self.staples_matrix = staples_matrix
        self.func = func

    def __call__(self, x):
        """Evaluate and return action."""
        return self.action(x)

    def action(self, x):
        """Return the action for the given input matrices."""
        if self.staples_matrix is not None:
            x = x @ self.staples_matrix

        if self.func is not None:
            x = self.func(x)

        reduced_trace = calc_reduced_trace(x)

        # Sum over trace, except on batch, if multi-point models are present
        if reduced_trace.ndim > 1:
            dim = tuple(range(1, reduced_trace.ndim))
            reduced_trace = torch.sum(reduced_trace, dim=dim)

        return -self.beta * torch.real(reduced_trace)

    def log_prob(self, x, action_logz=0):
        """Return log probability up to an additive constant."""
        return -self.action(x) - action_logz


def calc_reduced_trace(x):
    """Compute the reduced trace of x."""
    return torch.mean(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)
