# Copyright (c) 2021-2025 Javad Komijani

"""
Neural network blocks for lattice mean-field and power spectral density (PSD).

Includes:
    - PSDBlock_: combines mean-field and FFT-based PSD transformations.
    - Factory functions for constructing these networks.
"""

# pylint: disable=relative-beyond-top-level

from typing import Tuple
import torch
import numpy as np

from .._core import Module_
from .meanfield_ import make_meanfield_net
from .fftflow_ import make_fftnet


__all__ = ["make_psd_block", "PSDBlock_"]


def make_psd_block(
    lat_shape: Tuple[int, ...],
    ipsd_knots_len: int = 10,
    meanfield_n_layers: int = 0,
):
    """
    Build a PSDBlock_ with FFT-based and optional mean-field components.

    - If ``meanfield_n_layers > 0``:
        * MeanFieldNet_: handles the zero mode (lattice mean).
        * FFTNet_: handles non-zero modes (fluctuations), with the zero mode
          ignored.
    - If ``meanfield_n_layers == 0``:
        * Only FFTNet_ is constructed.

    Args:
        lat_shape: Lattice shape.
        ipsd_knots_len: Number of spline knots for FFTNet_ (> 2).
        meanfield_n_layers: Depth of the mean-field converter. If 0, no
            mean-field network is used. Default is 0.

    Returns:
        PSDBlock_ or FFTNet_: Combined PSD block or FFT-based network
    """
    if ipsd_knots_len <= 2:
        raise ValueError("ipsd_knots_len must be greater than 2")

    if meanfield_n_layers <= 0:
        # Only FFTNet_ is needed
        return make_fftnet(lat_shape, knots_len=ipsd_knots_len)

    # Build MeanFieldNet_ for zero mode
    mfnet_ = make_meanfield_net(meanfield_n_layers)

    # Build FFTNet_ for non-zero modes
    fftnet_ = make_fftnet(
        lat_shape, knots_len=ipsd_knots_len, ignore_zeromode=True
    )

    # Combine both components into a PSD block.
    return PSDBlock_(mfnet_=mfnet_, fftnet_=fftnet_)


class PSDBlock_(Module_):  # pylint: disable=invalid-name
    """
    Block that applies combined mean-field and FFT-based PSD transformations.

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
