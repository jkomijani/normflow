# Copyright (c) 2021-2024 Javad Komijani

"""This is a module for defining matrix models..."""


import torch


class MatrixAction:
    r"""The action is defined for $n \times n$ matrix M as

    .. math::
        S = \frac{\beta}{n} Tr (M G)

    where :math:`G` is a constant matrix that by default is set to identity.
    """

    def __init__(self, *, beta, staples_matrix=None):
        # staples_matrix is the constant `G` matrix above
        self.beta = beta
        self.staples_matrix = staples_matrix

    def __call__(self, cfgs):
        return self.action(cfgs)

    def action(self, cfgs):
        """Returns action corresponding to input configurations."""

        if self.staples_matrix is not None:
            cfgs = cfgs @ self.staples_matrix

        reduced_trace = calc_reduced_trace(cfgs).real

        # sum over the traces if it is a "more-than-one-point" matrix models.
        if reduced_trace.ndim > 1:
            dim = tuple(range(1, reduced_trace.ndim))  # 0 axis is batch axis
            reduced_trace = torch.sum(reduced_trace, dim=dim)

        return -self.beta * reduced_trace

    def log_prob(self, x, action_logz=0):
        """Returns log probability up to an additive constant."""
        return -self.action(x) - action_logz

    @property
    def parameters(self):
        return {'beta': self.beta}


def calc_trace(x):
    return torch.sum(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)


def calc_reduced_trace(x):  # reduced trace = 1/n trace()
    return torch.mean(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)
