# Copyright (c) 2025 Javad Komijani

"""
This module includes several basic subclasses of `torch.nn.Module` that are
designed specifically for scalar tensors. These subclasses implement various
particularly in probabilistic modeling and generative tasks.
"""

# pylint: disable=invalid-name, relative-beyond-top-level

from typing import Callable, Dict, Tuple
import math
import torch

from normflow.lib.spline import make_rq_spline_field

from .._core import Module_
from .time_embedding import TimeEmbeddedWeight


__all__ = [
    "RQSModule_",
    "TimeEmbeddedRQSModule_",
    "TimeEmbeddedUnityDistConvertor_",
    "TimeEmbeddedPhaseDistConvertor_"
]


# =============================================================================
class RQSModule_(Module_):
    """A module for learnable RQS-based transformations.

    This module defines a monotonic mapping from `xlim` to `ylim` using a
    trainable Rational Quadratic Spline (RQS) function. The first knot is fixed
    at (xlim[0], ylim[0]) and the last at (xlim[1], ylim[1]). Intermediate knot
    positions are either learned through the chosen `feature_map_fn`, or fixed
    if `knots_x` and/or `knots_y` are provided.

    The provided function `feature_map_fn` generates a feature map that is used
    to determine the coordinates of the intermediate knots (if not fixed), via
    a softmax, and derivatives at the knots (if `smooth=True`), via a softplus.

    Note that to represent n knots (i.e., n-1 spline segments), the feature map
    must produce `2 (n–1) + n` features in total: (n–1) for the x-coordinates,
    (n–1) for the y-coordinates, and n for the derivatives. This number is
    reduced when `knots_x` or `knots_y` are provided, or when `smooth=True`.

    When `xlim=(0, 1)` and `ylim=(0, 1)`, the transformation becomes a smooth
    bijection from [0, 1] to [0, 1], making it suitable for normalizing flows
    or differentiable coordinate transforms.

    Parameters
    ----------
    feature_map_fn : Callable
        Function producing features that parameterize the spline.
    xlim, ylim : tuple of float, optional
        Minimum and maximum values for x and y. Defaults to (0, 1).
    knots_x, knots_y : torch.Tensor or None, optional
        If provided, these fix the knot positions instead of learning them.
    knots_axis : int, optional
        Axis index used for interpreting the feature-map output. Default: -1.
    smooth : bool, optional
        If True, enforce smooth derivatives across knots. Default: True.
    extrap : dict, optional
        Extrapolation behavior outside the domain.
    """

    def __init__(
        self,
        feature_map_fn: Callable,
        xlim: Tuple[float, float] = (0, 1),
        ylim: Tuple[float, float] = (0, 1),
        knots_x: torch.Tensor = None,
        knots_y: torch.Tensor = None,
        knots_axis: int = -1,
        smooth: bool = True,
        extrap: Dict = None,
    ):
        super().__init__()
        self.spline_kwargs = {
            'xlim': xlim,
            'ylim': ylim,
            'knots_axis': knots_axis,
            'knots_x': knots_x,
            'knots_y': knots_y,
            'smooth': smooth,
            'extrap': extrap
        }
        self.feature_map_fn = feature_map_fn
        self.spline_shape = ()

    def forward(self, x: torch.Tensor, log0=0, args=None) -> torch.Tensor:
        """Compute the forward spline transformation.

        Args:
            x (torch.Tensor): Input tensor within the spline domain `xlim`.
            args (optional): Argument or tuple passed to `feature_map_fn`.
            log0 (torch.Tensor, float): Log-Jacobian of past transformations.

        Returns:
            torch.Tensor: Transformed tensor of the same shape.
        """
        # Build the spline transformation defined by current args
        spline = self.make_rq_spline_field(args)
        # Reshape input to match the spline shape & evaluate spline
        x_reshaped = x.reshape(*self.spline_shape, -1)
        y, g = spline(x_reshaped, grad=True)  # g is gradient @ x
        # Reshape outputs & calc total logj
        y, g = y.reshape(x.shape), g.reshape(x.shape)
        logj = self.sum_density(torch.log(g))
        return y, log0 + logj

    def reverse(self, y: torch.Tensor, log0=0, args=None) -> torch.Tensor:
        """Compute the inverse spline transformation.

        Args:
            y (torch.Tensor): Input tensor within the spline range `ylim`.
            args (optional): Argument or tuple passed to `feature_map_fn`.
            log0 (torch.Tensor, float): Log-Jacobian of past transformations.

        Returns:
            torch.Tensor: Inverse-transformed tensor of the same shape.
        """
        # Build the spline transformation defined by current args
        spline = self.make_rq_spline_field(args)
        # Reshape input to match the spline shape & evaluate reverse spline
        y_reshaped = y.reshape(*self.spline_shape, -1)
        x, g = spline.reverse(y_reshaped, grad=True)  # g is gradient @ x
        # Reshape outputs & calc total logj
        x, g = x.reshape(y.shape), g.reshape(y.shape)
        logj = self.sum_density(torch.log(g))
        return x, log0 + logj

    def make_rq_spline_field(self, args=None):
        """Constructs RQS with the saved key-word argumentes."""
        if not isinstance(args, tuple):
            args = () if args is None else (args,)

        feature_map = self.feature_map_fn(*args)
        if feature_map.shape[0] == 1:
            feature_map = feature_map.squeeze(0)  # squeeze the dummy batch dim

        self.spline_shape = feature_map.shape[:-1]
        return make_rq_spline_field(feature_map, **self.spline_kwargs)


# =============================================================================
class TimeEmbeddedRQSModule_(RQSModule_):
    """An RQS module whose spline parameters are conditioned on time.

    This class extends `RQSModule_` by generating spline parameters through a
    `TimeEmbeddedWeight` network. The time embedding produces features used to
    determine knot positions and/or derivatives in the spline.

    Args:
        n_knots (int): Number of spline knots.
        knots_x (torch.Tensor or None): Fixed x-knots, or learned if None.
        knots_y (torch.Tensor or None): Fixed y-knots, or learned if None.
        smooth (bool): If True, enforces first-derivative continuity.
        encoding_kwargs (dict): Args forwarded to `TimeEmbeddedWeight`.
        **rqs_kwargs: Additional args passed to `RQSModule_`.
    """
    def __init__(
        self,
        n_knots: int,
        knots_x: torch.Tensor | None = None,
        knots_y: torch.Tensor | None = None,
        smooth: bool = False,
        encoding_kwargs: Dict = None,
        **rqs_kwargs
    ):
        # Determine required feature dimensions
        n_x = (n_knots - 1) * (knots_x is None)
        n_y = (n_knots - 1) * (knots_y is None)
        n_d = n_knots * (not smooth)
        n_features = n_x + n_y + n_d

        # Time-conditioned feature mapping
        feature_map_fn = TimeEmbeddedWeight(
            [n_features], **(encoding_kwargs or {})
        )

        # Initialize the parent RQS module
        super().__init__(
            feature_map_fn,
            knots_x=knots_x,
            knots_y=knots_y,
            smooth=smooth,
            **rqs_kwargs
        )


# =============================================================================
class TimeEmbeddedUnityDistConvertor_(TimeEmbeddedRQSModule_):
    """As a PDF convertor for random variables in range [0, 1]."""

    def __init__(self, n_knots, symmetric=False, **kwargs):

        extra = {
            'xlim': (0.5, 1) if symmetric else (0, 1),
            'ylim': (0.5, 1) if symmetric else (0, 1),
            'extrap': {'left': 'anti'} if symmetric else None
        }

        super().__init__(n_knots, **kwargs, **extra)


# =============================================================================
class TimeEmbeddedPhaseDistConvertor_(TimeEmbeddedRQSModule_):
    """As a PDF convertor for random variables in range [-pi, pi]."""

    def __init__(self, n_knots, symmetric=False, **kwargs):

        pi = math.pi

        extra = {
            'xlim': (0, pi) if symmetric else (-pi, pi),
            'ylim': (0, pi) if symmetric else (-pi, pi),
            'extrap': {'left': 'anti'} if symmetric else None
        }

        super().__init__(n_knots, **kwargs, **extra)
