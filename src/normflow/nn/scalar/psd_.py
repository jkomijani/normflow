# Copyright (c) 2021-2025 Javad Komijani

"""
This module introduces a neural network to handle the PSD of a field.
"""

# pylint: disable=relative-beyond-top-level

import torch
import numpy as np

from .._core import Module_


class PSDBlock_(Module_):  # pylint: disable=invalid-name
    """Block that applies Mean-Field and FFT-based PSD transformations.

    The PSDBlock contains two components:
        1. MeanFieldNet_ for the zero-mode (mean) of the lattice.
        2. FFTNet_ for the non-zero modes (fluctuations).

    Note:
        If MeanFieldNet_ is used (i.e., the zero mode is handled),
        `ignore_zeromode` **must** be set to True in FFTNet_.

    Attributes:
        mfnet_ (MeanFieldNet_): Mean-field network.
        fftnet_ (FFTNet_): FFT-based network.
    """

    def __init__(self, *, mfnet_, fftnet_):
        """Initialize PSDBlock_.

        Args:
            mfnet_ (MeanFieldNet_): Mean-field network handling zero mode.
            fftnet_ (FFTNet_): FFT-based network for non-zero modes.
        """
        super().__init__()
        self.mfnet_ = mfnet_
        self.fftnet_ = fftnet_

    def forward(self, x, log0=0):
        """Forward pass through mean-field and FFTNet_.

        Args:
            x (torch.Tensor): Input lattice data.
            log0 (float | torch.Tensor): Base log-Jacobian. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Transformed lattice and updated log-Jacobian.
        """
        dim = list(range(1, x.dim()))
        rvol = np.prod(x.shape[1:]) ** 0.5  # sqrt of volume
        x_mean = torch.mean(x, dim=dim, keepdim=True)

        y_mf, logj_mf = self.mfnet_.forward(x_mean, rvol=rvol)
        y_fft, logj_fft = self.fftnet_.forward(x - x_mean)
        return (y_mf + y_fft), (log0 + logj_mf + logj_fft)

    def reverse(self, x, log0=0):
        """Reverse pass through mean-field and FFTNet_.

        Args:
            x (torch.Tensor): Input lattice data.
            log0 (float | torch.Tensor): Base log-Jacobian. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Reconstructed lattice and updated log-Jacobian.
        """
        dim = list(range(1, x.dim()))
        rvol = np.prod(x.shape[1:]) ** 0.5  # sqrt of volume
        x_mean = torch.mean(x, dim=dim, keepdim=True)

        y_mf, logj_mf = self.mfnet_.reverse(x_mean, rvol=rvol)
        y_fft, logj_fft = self.fftnet_.reverse(x - x_mean)
        return (y_mf + y_fft), (log0 + logj_mf + logj_fft)

    def _hack(self, x, log0=0):
        """Forward pass with returning intermediate components."""
        dim = list(range(1, x.dim()))
        rvol = np.prod(x.shape[1:]) ** 0.5  # sqrt of volume
        x_mean = torch.mean(x, dim=dim, keepdim=True)

        y_mf, logj_mf = self.mfnet_.forward(x_mean, rvol=rvol)
        y_fft, logj_fft = self.fftnet_.forward(x - x_mean)
        stack = [
            (x_mean, log0),
            (y_mf, logj_mf),
            (y_fft, logj_fft),
            ((y_mf + y_fft), (log0 + logj_mf + logj_fft))
        ]
        return stack

    def transfer(self, **kwargs) -> 'PSDBlock_':
        """Return a new PSDBlock_ with transferred submodules."""
        return self.__class__(
            mfnet_=self.mfnet_.transfer(**kwargs),
            fftnet_=self.fftnet_.transfer(**kwargs)
        )
