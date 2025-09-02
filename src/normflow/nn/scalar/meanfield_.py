# Copyright (c) 2021-2025 Javad Komijani

"""
This module introduces a neural network to handle the mean field of a field.
"""

# pylint: disable=relative-beyond-top-level

import torch
import numpy as np

from .modules_ import DistConvertor_
from .._core import Module_


__all__ = ["make_meanfieldnet", "MeanFieldNet_"]


def make_meanfieldnet(knots_len: int, **kwargs) -> 'MeanFieldNet_':
    """
    Build a MeanFieldNet_ with an internal DistConvertor_.

    Parameters
     ----------
    knots_len : int
        Number of knots for the internal converter. Defaults to 10.
    **kwargs : dict, optional
        Additional arguments passed to DistConvertor_.

    Returns:
        MeanFieldNet_: Constructed mean-field network.
    """
    dc_ = DistConvertor_(knots_len, **kwargs)
    return MeanFieldNet_(dc_)


class MeanFieldNet_(Module_):  # pylint: disable=invalid-name
    """Mean-Field Network for transforming the spatial mean of a lattice.

    This module converts the mean-field part of a lattice using an internal
    distribution converter. It normalizes the mean by the square root of
    the spatial volume to maintain proper scaling.

    Attributes:
        dc_ (DistConvertor_):
            Internal distribution converter acting on the mean-field.
    """

    def __init__(self, dc_: 'DistConvertor_'):
        super().__init__()
        self.dc_ = dc_

    def forward(self, x, log0=0, rvol=None):
        """Forward mean-field transformation of lattice data.

        The lattice mean is scaled by the square root of the spatial volume,
        transformed by `dc_`, then rescaled.

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
            x_mean_new_scaled, log0 = self.dc_.forward(x_mean * rvol, log0)
            # Replace original mean with transformed mean, keep fluctuations
            return x + (x_mean_new_scaled / rvol - x_mean), log0
        else:
            # Mean-field convention: transform already-computed mean
            x_mean_new_scaled, log0 = self.dc_.forward(x * rvol, log0)
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
            x_mean_new_scaled, log0 = self.dc_.reverse(x_mean * rvol, log0)
            # Replace original mean with transformed mean, keep fluctuations
            return x + (x_mean_new_scaled / rvol - x_mean), log0
        else:
            # Mean-field convention: transform already-computed mean
            x_mean_new_scaled, log0 = self.dc_.reverse(x * rvol, log0)
            return x_mean_new_scaled / rvol, log0

    def _hack(self, x, log0=0):
        """Forward pass with returning intermediate mean-field components."""
        dim = list(range(1, x.dim()))
        rvol = np.prod(x.shape[1:]) ** 0.5
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        stack = [(x_mean.ravel(), log0)]
        x_mean_scaled, log0 = self.dc_.forward(x_mean * rvol, log0)
        stack.append((x_mean_scaled.ravel() / rvol, log0))
        return stack

    @staticmethod
    def build(*args, **kwargs):
        """For legacy; will be removed in future."""
        return make_meanfieldnet(*args, **kwargs)
