# Copyright (c) 2021-2025 Javad Komijani

"""
This module defines neural network components for handling the mean-field part
of lattice data, as motivated by mean-field theory.

Includes:
    - MeanFieldNet_: transforms the lattice mean with proper scaling.
    - make_meanfield_net: builds a MeanFieldNet_ using Pade32List_.
    - Pade32List_: sequence of Pade32_ layers.
    - MeanFieldNet_.build: legacy constructor (deprecated).
"""

# pylint: disable=relative-beyond-top-level

import torch
import numpy as np

from .modules_ import DistConvertor_
from .modules_ import Pade32_
from .._core import Module_
from .._core import ModuleList_


__all__ = ["make_meanfield_net", "MeanFieldNet_"]


def make_meanfield_net(n: int) -> 'MeanFieldNet_':
    """
    Construct a MeanFieldNet_ with a Pade32List_ converter.

    - Pade32List_: stack of odd functions, suitable for Z2-symmetric theories.
    - For other cases, one could use Pade32a or DistConvertor_,
      which are not supported here.

    Args:
        n: Number of Pade32_ layers.

    Returns:
        MeanFieldNet_: Constructed mean-field network.
    """
    net_ = Pade32List_(n)
    return MeanFieldNet_(net_)


class MeanFieldNet_(Module_):  # pylint: disable=invalid-name
    """Mean-Field Network for transforming the spatial mean of a lattice.

    This module converts the mean-field part of a lattice using an internal
    distribution converter. It normalizes the mean by the square root of
    the spatial volume to maintain proper scaling.

    Attributes:
        net_ (DistConvertor_):
            Internal distribution converter acting on the mean-field.
    """

    def __init__(self, net_: 'Module_'):
        super().__init__()
        self.net_ = net_

    def forward(self, x, log0=0, rvol=None):
        """Forward mean-field transformation of lattice data.

        The lattice mean is scaled by the square root of the spatial volume,
        transformed by `net_`, then rescaled.

        The method supports two conventions:

        1. Full lattice convention (rvol is None):
           - `x` contains the full lattice field.
           - The mean over the spatial dimensions is computed and transformed.
           - The output has the same shape as `x`, with the mean replaced
             by the transformed mean and the fluctuations preserved.

        2. Mean-field convention (rvol is provided):
           - `x` already represents the lattice mean.
           - `rvol` is the square root of the lattice volume.
           - Only the provided mean is transformed, and the output shape
             matches `x`.

        Args:
            x (torch.Tensor): Input lattice data.
            log0 (float | torch.Tensor): Base log-Jacobian. Defaults to 0.
            rvol (float, optional): Precomputed sqrt of lattice volume.
                If None, assume the input x contains the full lattice field.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Transformed lattice and updated log-Jacobian.
        """
        if rvol is None:
            # Full lattice: compute mean across spatial dimensions
            dim = list(range(1, x.dim()))
            rvol = np.prod(x.shape[1:]) ** 0.5  # sqrt of lattice volume
            # Calculate and transform the mean
            x_mean = torch.mean(x, dim=dim, keepdim=True)
            x_mean_new_scaled, log0 = self.net_.forward(x_mean * rvol, log0)
            # Replace original mean with transformed mean, keep fluctuations
            return x + (x_mean_new_scaled / rvol - x_mean), log0
        else:
            # Mean-field convention: transform already-computed mean
            x_mean_new_scaled, log0 = self.net_.forward(x * rvol, log0)
            return x_mean_new_scaled / rvol, log0

    def reverse(self, x, log0=0, rvol=None):
        """Reverse mean-field transformation.

        Args:
            x (torch.Tensor): Input lattice data.
            log0 (float | torch.Tensor): Base log-Jacobian. Defaults to 0.
            rvol (float, optional): Precomputed sqrt of lattice volume.
                If None, assume the input x contains the full lattice field.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Reconstructed lattice and updated log-Jacobian.
        """
        if rvol is None:
            # Full lattice: compute mean across spatial dimensions
            dim = list(range(1, x.dim()))
            rvol = np.prod(x.shape[1:]) ** 0.5  # sqrt of lattice volume
            # Calculate and transform the mean
            x_mean = torch.mean(x, dim=dim, keepdim=True)
            x_mean_new_scaled, log0 = self.net_.reverse(x_mean * rvol, log0)
            # Replace original mean with transformed mean, keep fluctuations
            return x + (x_mean_new_scaled / rvol - x_mean), log0
        else:
            # Mean-field convention: transform already-computed mean
            x_mean_new_scaled, log0 = self.net_.reverse(x * rvol, log0)
            return x_mean_new_scaled / rvol, log0

    def _hack(self, x, log0=0):
        """Forward pass with returning intermediate mean-field components."""
        dim = list(range(1, x.dim()))
        rvol = np.prod(x.shape[1:]) ** 0.5
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        stack = [(x_mean.ravel(), log0)]
        x_mean_scaled, log0 = self.net_.forward(x_mean * rvol, log0)
        stack.append((x_mean_scaled.ravel() / rvol, log0))
        return stack

    @staticmethod
    def build(knots_len=10, **kwargs):
        """
        Legacy constructor for `MeanFieldNet_`.

        This method creates a `MeanFieldNet_` using a `DistConvertor_`.

        !!! warning
            Deprecated: This method will be removed in future releases.
            Use :func:`make_meanfield_net` instead, which builds the
            internal converter using :class:`Pade32List_` layers.
        """
        dc_ = DistConvertor_(knots_len, **kwargs)
        return MeanFieldNet_(dc_)


class Pade32List_(ModuleList_):  # pylint: disable=invalid-name
    """A sequence of instances of `Pade32_`."""
    def __init__(self, n: int):
        super().__init__([Pade32_() for _ in range(n)])
