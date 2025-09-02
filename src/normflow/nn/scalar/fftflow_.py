# Copyright (c) 2021-2025 Javad Komijani

r"""This module introduces a new neural network called `FFTNet_`.

Theory: The Effective Action
--------------------------
[e.g. Shrednicki's book, chapter 21]

We define the effective action

.. math:

    \Gamma[\phi] S = \int d^n k (
        \tilde \phi(-k) (\kappa k^2 + m^2 - \Pi(k^2)) \tilde \phi(k) + \cdots
        ).

where :math:`\tilde \phi(k)` is the Fourier transform of :math:`\phi(x)`.
The effective action has the property that the tree-level Feynman diagram that
it generates reproduces the complete scattering amplitude of the original
theory.
"""

# pylint: disable=relative-beyond-top-level, disable=not-callable

from typing import Tuple
import copy
import torch
import numpy as np

from .modules import SplineNet
from .._core import Module_
from ...lib.indexing import outer_arange


irfft, rfft = torch.fft.irfftn, torch.fft.rfftn

__all__ = ["make_fftnet", "FFTNet_"]


# =============================================================================
def make_fftnet(
    lat_shape: Tuple[int],
    knots_len: int,
    eff_mass2: float = 1,
    eff_kappa: float = 1,
    a: float = 1,
    **ipsd_kwargs
) -> "FFTNet_":
    """
    Construct an FFTNet_ with a default IPSD multiplier network.

    Parameters
     ----------
    lat_shape : Tuple[int]
        Lattice shape.
    knots_len : int
        Number of spline knots for IPSDMultiplierNet.
    eff_mass2 : float, optional
        Effective mass squared for initial IPSD scaling. Default is 1.
    eff_kappa : float, optional
        Effective factor for k^2 scaling. Default is 1.
    a : float, optional
        Overall scaling factor for initial IPSD coefficients. Default is 1.
    **ipsd_kwargs : dict, optional
        Additional arguments for IPSDMultiplierNet constructor.

    Returns
    -------
    FFTNet_
        Constructed FFT network with IPSD scaling.
    """
    max_lat_k2 = torch.max(calc_lattice_k2(lat_shape))

    logm2 = torch.log(torch.tensor(eff_mass2))
    logk2 = torch.log(eff_kappa * max_lat_k2)

    logy = IPSDMultiplierNet.apply_scale(
        torch.tensor([logm2, logk2]), a=a, ndim=len(lat_shape)
    )

    ipsd_multiplier_net = IPSDMultiplierNet(knots_len, logy, **ipsd_kwargs)
    return FFTNet_(lat_shape, ipsd_multiplier_net)


# =============================================================================
class FFTNet_(Module_):  # pylint: disable=invalid-name
    r"""
    FFT-based network with optional IPSD scaling.

    This network applies an elementwise multiplier in Fourier space to scale
    lattice modes according to an inverse power spectral density (IPSD). It
    supports input data with or without batch axes.

    - Examples:

         >>> net = normflow.FFTNet_((4, 4))
         >>> prior = normflow.NormalPrior(shape=(4, 4))
         >>> samples = prior.sample(13)
         >>> samples.shape
         torch.Size([13, 4, 4])
         >>> net(samples).shape
         torch.Size([13, 4, 4])
         >>> net(samples[0]).shape
         torch.Size([4, 4])

    Parameters
    ----------
    lat_shape : Tuple[int]
        Lattice shape for FFT dimensions.
    ipsd_multiplier_net : IPSDMultiplierNet
        Network that provides the inverse PSD multiplicative factor.
    """

    def __init__(
        self, lat_shape: Tuple[int],
        ipsd_multiplier_net: 'IPSDMultiplierNet'
    ):
        super().__init__()
        self.lat_ndim = len(lat_shape)
        self.lat_shape = lat_shape
        self.ipsd_multiplier_net = ipsd_multiplier_net

        # Define FFT dimensions using negative indices to handle optional batch
        self.rfft_dim = list(range(-self.lat_ndim, 0))
        self.rfft_axis = -1  # axis reduced by rfft for redundancy

        # Precompute normalized lattice k^2 for IPSD scaling
        lat_k2 = calc_lattice_k2(self.lat_shape)
        self.register_buffer('norm_lat_k2', lat_k2 / torch.max(lat_k2))
        self.register_buffer('max_lat_k2', torch.max(lat_k2))

    def forward(self, x, log0=0):
        r"""
        Forward transform: scale lattice modes in Fourier space.

        Applies

        .. math::
            y = \mathcal{F}^{-1}\big[ \mathcal{F}[x] \cdot w \big],

        where :math:`w = 1 / \sqrt{\sigma(k^2)}` is the IPSD multiplier.

        Args:
            x (torch.Tensor): Input lattice data.
            log0 (float | torch.Tensor): Base log-Jacobian. Default is 0.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                Transformed lattice and updated log-Jacobian.
        """
        w = 1 / self.ipsd_multiplier**0.5
        logj = self.log_jacobian(w)
        dim = self.rfft_dim
        return irfft(rfft(x, dim=dim) * w, dim=dim), log0 + logj

    def reverse(self, x, log0=0):
        r"""
        Reverse transform: invert IPSD scaling in Fourier space.

        Applies

        .. math::
            x = \mathcal{F}^{-1}\big[ \mathcal{F}[y] / w \big],

        Args:
            x (torch.Tensor): Input lattice data.
            log0 (float | torch.Tensor): Base log-Jacobian. Default is 0.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                Transformed lattice and updated log-Jacobian.
        """
        w = 1 / self.ipsd_multiplier**0.5
        logj = self.log_jacobian(w)
        dim = self.rfft_dim
        return irfft(rfft(x, dim=dim) / w, dim=dim), log0 - logj

    @property
    def ipsd_multiplier(self):
        r"""
        Compute the lattice IPSD multiplicative factor :math:`\sigma(k^2)`.

        Returns:
            torch.Tensor: The inverse PSD multiplicative factor.
        """
        return self.ipsd_multiplier_net(self.norm_lat_k2)

    @staticmethod
    def build(*args, **kwargs):
        """For legacy; will be removed in future."""
        return make_fftnet(*args, **kwargs)

    def log_jacobian(self, weights):
        """
        Compute the logarithm of the Jacobian for multiplication by `weights`.

        Notes
        -----
        - The FFT itself has unit Jacobian, so only the elementwise weights
          contribute.
        - For rfftn, each positive k-mode has a symmetric negative k-mode,
          which doubles their contribution. Hence the factor 2 below.
        - The modes k=0 and k=pi/a do not have a symmetric pair, so their
          contributions are handled separately.

        Args:
            weights (torch.Tensor): Fourier-space multiplicative weights.

        Returns:
            torch.Tensor: Log-Jacobian tensor.
        """
        def sumlog(w):
            return torch.sum(torch.log(w), dim=self.rfft_dim)

        w = weights
        logj = 2 * sumlog(w)  # account for symmetric positive/negative k-modes
        # subtract contributions of k=0 and k=pi/a, which are unique
        logj -= (sumlog(w[..., 0:1]) + sumlog(w[..., -1:]))

        return logj

    @property
    def infrared_mass(self):
        """Dimension-less mass (in lattice units)"""
        return self.ipsd_net.infrared_mass(self.max_lat_k2)

    def transfer(self, scale_factor=1, shape=None, **extra):
        """Map the weights of the current lattice to a new lattice.

        Parameters
        ----------
        shape : tuple of integers
            The shape of the lattice ...

        scale_factor : float
            The factor for improving the resolution; inverse of ratio of the
            lattice spacing of the new lattice compared to the current one.

        **How to set the scale_factor**:
        For example, for converting a netwrok corresponding to `a = 2` fm to
        another one corresponding to `a = 0.2` fm, set scale_factor to 10.
        """
        shape = self.lat_shape if shape is None else shape
        ipsd_multiplier_net = self.ipsd_multiplier_net.transfer(
            scale_factor=scale_factor, ndim=self.lat_ndim
        )
        return self.__class__(shape, ipsd_multiplier_net=ipsd_multiplier_net)


# =============================================================================
class IPSDMultiplierNet(SplineNet):
    r"""
    Inverse Power Spectral Density (IPSD) multiplicative factor.

    This class produces a multiplicative factor to scale the PSD of a signal
    in Fourier/lattice space according to a target inverse PSD. The input
    tensor represents **lattice momentum squared values** (:math:`k^2`).

    The factor can be applied to **any PSD**, not just white noise. In the
    special case where the input is white Gaussian noise, whose PSD is flat:

    .. math::
        \text{PSD}_{\rm white}(k) = \text{const},

    applying this multiplier modifies the inverse PSD:

    .. math::
        \tilde{\eta}(k) \mapsto \frac{\tilde{\eta}(k)}{\sigma(k^2)},\quad
        \text{PSD}_{\rm scaled}(k) \propto \frac{1}{\sigma(k^2)^2},

    so the noise becomes **colored** according to the specified scaling.

    The factor itself is modeled as:

    .. math::
        \sigma(k^2)^2 = y_0 + y_1 \, f_\text{spline}(k^2),

    where :math:`f_\text{spline}` is a **monotonically increasing rational-
    quadratic (RQ) spline** defined by the base ``SplineNet``, and
    ``y = exp(logy)`` are learnable parameters. The monotonicity ensures that
    larger lattice k^2 generally correspond to larger scaling factors, which
    is physically reasonable for inverse PSDs.

    Parameters
    ----------
    knots_len : int
        Number of spline knots used in the base SplineNet. If less than 2,
        it is automatically set to 2 and `smooth=True` is added to `kwargs`,
        which effectively makes the spline behave like an identity function.
    logy : torch.Tensor
        Initial logarithm of the coefficients :math:`y = [y_0, y_1]`.
        Registered as a learnable parameter.
    ignore_zeromode : bool, optional
        If True, the zero lattice mode (:math:`k^2 = 0`) is replaced with 1
        to prevent singularities in Jacobian computations. Default is False.
    **kwargs : dict
        Additional arguments forwarded to the ``SplineNet`` constructor.
    """

    def __init__(
        self,
        knots_len: int,
        logy: torch.Tensor,
        ignore_zeromode: bool = False,
        **kwargs
    ):
        if knots_len < 2:
            knots_len = 2
            kwargs.update({'smooth': True})
            # with these commands, the base spline class behaves like identity

        super().__init__(knots_len, **kwargs)
        self.logy = torch.nn.Parameter(logy)
        self.ignore_zeromode = ignore_zeromode

    def forward(self, k2: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the IPSD multiplicative factor for lattice k^2 values.

        Computes

        .. math::
            \sigma(k^2)^2 = y_0 + y_1 \, f_\text{spline}(k^2)

        Parameters
        ----------
        k2 : torch.Tensor
            Input tensor of lattice momentum squared values.

        Returns
        -------
        torch.Tensor
            Tensor of IPSD scaling factors of the same shape as `k2`.
            If ``ignore_zeromode=True``, the zero lattice mode is set to 1.
        """
        y = torch.exp(self.logy)
        sigma_k2 = y[0] + y[1] * super().forward(k2)

        if self.ignore_zeromode:
            # Replace the zero mode (all indices 0) with 1
            sigma_k2[(0,) * k2.dim()] = 1

        return sigma_k2

    def transfer(self, scale_factor=1, ndim=1):
        """Map the weights of the current lattice to a new lattice."""
        ipsd = copy.deepcopy(self)
        state_dict = ipsd.state_dict()
        logy = self.apply_scale(
            state_dict['logy'], a=1/scale_factor, ndim=ndim
        )
        state_dict.update({'logy': logy})
        ipsd.load_state_dict(state_dict)
        return ipsd

    @staticmethod
    @torch.no_grad()
    def apply_scale(logy, *, a, ndim):
        """Scale `m^2` and `k^2` as the lattice spacing `a` changes."""
        a = torch.tensor(a)
        logm2 = logy[0] + torch.log(a) * ndim
        logk2 = logy[1] + torch.log(a) * (ndim - 2)
        return torch.tensor([logm2, logk2])

    @torch.no_grad()
    def infrared_mass(self, max_lat_k2):
        """Compute the effective infrared mass."""
        return torch.exp(0.5 * self.logy[0])


# =============================================================================
def calc_lattice_k2(lat_shape: Tuple[int]) -> torch.Tensor:
    r"""
    Compute the lattice momentum-squared grid, trimmed for rFFT.

    For each lattice axis of size n, the momentum-space values are defined on
    the interval :math:`[0, 2\pi (1 - 1/n)]` with n discrete points. These 1D
    grids are combined into a multi-dimensional lattice :math:`k^2` array
    using :func:`outer_lattice_k2`.

    The lattice dispersion relation is defined as

    .. math::
        k^2_{\text{lat}}(\mathbf{k})
        = \sum_{i=1}^d 4 \sin^2 \left(\frac{k_i}{2}\right),

    where :math:`\mathbf{k} = (k_1, \ldots, k_d)` are the discretized momenta.

    Since real-valued FFTs (``torch.fft.rfftn``) only store the positive half
    of the last frequency axis, the result is trimmed along the last axis to
    length :math:`1 + n//2`.

    Args:
        lat_shape (Tuple[int]): Lattice dimensions (e.g., (Nx, Ny, Nz)).

    Returns:
        torch.Tensor: Multi-dimensional array of lattice k^2 values with
        the last axis trimmed for rFFT compatibility.
    """
    momentum_layout = tuple((0., 2 * np.pi * (1 - 1/n), n) for n in lat_shape)
    lat_k2 = outer_lattice_k2(momentum_layout)
    lat_k2 = lat_k2[..., :(1 + lat_shape[-1]//2)]  # trim for rfftn
    return lat_k2


def outer_lattice_k2(momentum_layout: Tuple[Tuple, ...]) -> torch.Tensor:
    r"""
    Generate a multi-dimensional lattice momentum-squared grid.

    This constructs a tensor representing

    .. math::
        k^2_{\text{lat}}(\mathbf{k})
        = \sum_{i=1}^d 4 \sin^2 \left(\frac{k_i}{2}\right),

    where :math:`\mathbf{k} = (k_1, \ldots, k_d)` are the discretized momenta
    along each lattice axis. It wraps :func:`outer_arange` but modifies:

      1. ``arange_gen`` to compute :math:`4 \sin^2(k/2)` values.
      2. ``rule`` to sum values across dimensions.

    Args:
        momentum_layout (Tuple[Tuple[float, float, int], ...]): For each axis,
            ``(start, stop, steps)`` defining the discretization range.

    Returns:
        torch.Tensor: Multi-dimensional array of summed lattice :math:`k^2`.

    Example:
        >>> outer_lattice_k2(tuple([(0, 1, 3) for _ in range(2)]))
        tensor([[0.0000, 0.2448, 0.9194],
                [0.2448, 0.4897, 1.1642],
                [0.9194, 1.1642, 1.8388]])
    """
    def arange_gen(*k_tuple: float) -> torch.Tensor:
        def lat_k2(k):
            return 4 * torch.sin(k/2) ** 2  # lattice k^2
        return lat_k2(torch.linspace(*k_tuple))

    return outer_arange(
        momentum_layout, rule=lambda a, b: a + b, arange_gen=arange_gen
    )
