# Copyright (c) 2025-2026 Javad Komijani

"""
This module includes several basic subclasses of `torch.nn.Module` that are
designed specifically for scalar tensors. These subclasses implement various
particularly in probabilistic modeling and generative tasks.
"""

# pylint: disable=invalid-name, relative-beyond-top-level

from typing import Callable, Dict, Tuple
import torch

from normflow.lib.spline import make_rq_spline_field

from .._core import ModuleList_, Module_

from .modules_ import Expit_, Logit_


__all__ = [
    "RQSplineContextModule_",
    "make_real_line_rqs_context_module",
]


# =============================================================================
class RQSplineContextModule_(Module_):
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
    xlim, ylim : tuple of float or None, optional
        Minimum and maximum values for x and y. Defaults to (0, 1) if not
        provided (`None`).
    knots_x, knots_y : torch.Tensor or None, optional
        If provided, these fix the knot positions instead of learning them.
    knots_axis : int, optional
        Axis index used for interpreting the feature-map output. Default: -1.
    smooth : bool, optional
        If True, enforce smooth derivatives across knots. Default: True.
    extrap : dict, optional
        Extrapolation behavior outside the domain.
    symmetric : bool, optional
        If True, restricts the spline to `xlim=(0.5, 1)`, `ylim=(0.5, 1)`,
        with an anti-periodic boundary condition (`extrap={'left': 'anti'}`)
        extending it to the left -- halving the number of features needed
        to cover [0, 1]. Defaults to False. Mutually exclusive with passing
        `xlim`/`ylim`/`extrap` explicitly (raises `TypeError` if both are
        given).
    """

    def __init__(
        self,
        feature_map_fn: Callable,
        xlim: Tuple[float, float] | None = None,
        ylim: Tuple[float, float] | None = None,
        knots_x: torch.Tensor = None,
        knots_y: torch.Tensor = None,
        knots_axis: int = -1,
        smooth: bool = True,
        extrap: Dict = None,
        symmetric: bool = False,
    ):
        if symmetric:
            if xlim is not None or ylim is not None or extrap is not None:
                raise TypeError(
                    "symmetric=True is mutually exclusive with explicitly"
                    " passing xlim/ylim/extrap"
                )
            xlim, ylim, extrap = (0.5, 1), (0.5, 1), {'left': 'anti'}
        else:
            xlim = (0, 1) if xlim is None else xlim
            ylim = (0, 1) if ylim is None else ylim

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
def make_real_line_rqs_context_module(
    feature_map_fn: Callable,
    **kwargs
) -> ModuleList_:
    """
    Build a context-conditioned RQ-spline-based transformation for
    unbounded, real variables.

    Steps:
        pass through instances of `Expit_`, `RQSplineContextModule_`, and
        `Logit_`.

    Parameters
    ----------
    feature_map_fn : Callable
        Forwarded to `RQSplineContextModule_`.
    **kwargs
        Additional keyword arguments forwarded to `RQSplineContextModule_`
        (e.g. `channels_axis`, `xlim`, `ylim`, `smooth`, `extrap`).
    """

    nets_ = [
        Expit_(), RQSplineContextModule_(feature_map_fn, **kwargs), Logit_(),
    ]
    return ModuleList_(nets_)
