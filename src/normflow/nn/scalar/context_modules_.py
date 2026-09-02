# Copyright (c) 2026 Javad Komijani

"""
This module provides subclasses of `Module_` that perform transformations
conditioned on an externally supplied context. These transformations involve
two sequential stages:

1. **First Transformation (on the Context)**:
   This stage uses a callable (typically an instance of `torch.nn.Module`
   with trainable parameters) to process the context tensor(s). The output
   from this transformation is then passed as parameters to the second stage.

2. **Second Transformation (on the Input)**:
   In this stage, the input is transformed using the parameters generated
   from the first stage. This transformation is an instance of `Module_`.

Unlike the mask-based coupling layers in `couplings_.py` -- where the frozen
half of the input comes from splitting the very tensor being transformed --
the context here is an independently supplied tensor. It is passed through
the `args` parameter of `forward`/`reverse`.

The trailing underscore in `Module_` and its subclasses indicates that the
`forward` and `reverse` methods return a two-item tuple:
   1. The transformed input.
   2. The logarithm of the Jacobian determinant, which is essential in cases
      like normalizing flows or transformations that involve volume changes.
"""

# pylint: disable=invalid-name, relative-beyond-top-level

from abc import abstractmethod, ABC
from typing import Callable, Dict, Tuple

import torch

from .._core import Module_

from ...lib.spline import make_rq_spline_field

from .modules_ import Affine_
from .modules_ import Pade32a_


__all__ = [
    "ContextModule_",
    "AffineContextModule_",
    "Pade32aContextModule_",
]


Tensor = torch.Tensor


# =============================================================================
class ContextModule_(Module_, ABC):
    """
    An abstract subclass of `Module_` designed for creating invertible
    transformations conditioned on an externally supplied context, involving
    two sequential stages:

    1. **First Transformation (on the Context)**:
       This stage uses a callable (typically an instance of `torch.nn.Module`
       with trainable parameters) to process the context tensor(s).
       The output from this transformation is then passed as parameters to
       the second stage.

    2. **Second Transformation (on the Input)**:
       In the second stage, the input is transformed using the parameters
       generated from the first stage. This transformation is an instance
       of `Module_`.

    The callable processing the context (`feature_map_fn`, typically an
    instance of `torch.nn.Module`) must be provided during instantiation.
    The context itself is supplied at call time via the `args` parameter
    of `forward`/`reverse` and can be a single tensor, a tuple of several
    tensors or no tensor (`None`).

    This class is named `ContextModule_` because its transformations are
    conditioned on an externally supplied context.
    """

    def forward(
        self,
        x: Tensor,
        log0: Tensor | float = 0,
        args: Tensor | Tuple[Tensor, ...] | None = None,
    ):
        """Apply the context-conditioned forward transformation.

        Args:
            x: The input to be transformed.
            log0: Log-Jacobian accumulated from previous transformations.
                Defaults to 0.
            args: The conditioning input(s) passed to `feature_map_fn`; see
                `_as_args_tuple` for the accepted shapes.

        Returns:
            Tensor: The transformed input.
            Tensor | float: Updated log-Jacobian.
        """
        conditioned_transform_ = self.make_conditioned_transform(args)
        return conditioned_transform_.forward(x, log0=log0)

    def reverse(
        self,
        x: Tensor,
        log0: Tensor | float = 0,
        args: Tensor | Tuple[Tensor, ...] | None = None,
    ):
        """Apply the context-conditioned reverse transformation.

        For the transformation to invert correctly, `args` must match the value
        used in the corresponding forward call; see `forward` for the meaning
        of the other arguments.

        Returns:
            Tensor: The inverse-transformed input.
            Tensor | float: Updated log-Jacobian.
        """
        conditioned_transform_ = self.make_conditioned_transform(args)
        return conditioned_transform_.reverse(x, log0=log0)

    @staticmethod
    def _as_args_tuple(args):
        """
        Normalize `args` into a tuple to splat into `feature_map_fn`.
        """
        if isinstance(args, tuple):
            return args
        return () if args is None else (args,)

    @abstractmethod
    def make_conditioned_transform(self, args=None) -> Module_:
        """Build a transformation, conditioned on `args`.

        Subclasses process `args` through their `feature_map_fn` and use
        its output to construct and return a `Module_` instance.

        Args:
            args: The conditioning input(s); see `_as_args_tuple`.

        Returns:
            Module_: The transformation to apply to `x`.
        """


# =============================================================================
class AffineContextModule_(ContextModule_):
    """
    A wrapper for `Affine_` that applies an affine transformation of the form
    :math:`a x + b`, where the parameters :math:`a` (scaling factor) and
    :math:`b` (bias term) are dynamically computed from a context tensor.
    The module is named `AffineContextModule_` because the affine
    transformation is conditioned on this context tensor.

    At instantiation, this module accepts a callable (typically an instance
    of `torch.nn.Module`), which is responsible for transforming the context
    tensor to a new tensor with two channels that will be used as weights for
    the scaling factor (`w_scale`) and bias term (`w_bias`) in the underlying
    `Affine_` module to perform the affine transformation.

    Parameters
    ----------
    feature_map_fn : Callable
        Function (or `torch.nn.Module`) that maps the context to a
        two-channel feature tensor, split into (`w_scale`, `w_bias`).
    channels_axis : int, optional
        Axis of `feature_map_fn`'s output along which the two channels are
        split. Default: -1.
    """

    def __init__(self, feature_map_fn: Callable, channels_axis: int = -1):
        super().__init__()
        self.feature_map_fn = feature_map_fn
        self.channels_axis = channels_axis

    def make_conditioned_transform(self, args=None) -> Module_:
        """
        Build an `Affine_` whose (`w_scale`, `w_bias`) come from:

            out = feature_map_fn(args)
        """
        out = self.feature_map_fn(*self._as_args_tuple(args))
        w_scale, w_bias = torch.tensor_split(out, 2, self.channels_axis)
        return Affine_(w_scale=w_scale, w_bias=w_bias)


# =============================================================================
class Pade32aContextModule_(ContextModule_):
    r"""
    A wrapper for `Pade32a_` that applies a Pade approximant of order [3/2]

    .. math::

        f(x) = a z \frac{a + z^2}{1 + a z^2}

    where :math:`z = s x + b`.
    (For details of the transformation see `Pade32a_`.)
    Parameters :math:`s, b, a` are dynamically computed from a context tensor.
    The module is named `Pade32aContextModule_` because the `pade32a_`
    transformation is conditioned on the context tensor.

    At instantiation, this module accepts a callable (typically an instance
    of `torch.nn.Module`), which is responsible for transforming the context
    tensor to a new tensor with three channels that will be used as weights for
    the scaling factor (`w_scale`), bias term (`w_bias`), and :math:`a` (`w_a`)
    in the underlying `Pade32a_` module.

    Parameters
    ----------
    feature_map_fn : Callable
        Function (or `torch.nn.Module`) that maps the context to a
        three-channel feature tensor, split into
        (`w_scale`, `w_bias`, `w_a`).
    channels_axis : int, optional
        Axis of `feature_map_fn`'s output along which the three channels
        are split. Default: -1.
    """

    def __init__(self, feature_map_fn: Callable, channels_axis: int = -1):
        super().__init__()
        self.feature_map_fn = feature_map_fn
        self.channels_axis = channels_axis

    def make_conditioned_transform(self, args=None) -> Module_:
        """
        Build an `Pade32a_` whose (`w_scale`, `w_bias`, `w_a`) come from:

            out = feature_map_fn(args)
        """
        out = self.feature_map_fn(*self._as_args_tuple(args))
        w_scale, w_bias, w_a = torch.tensor_split(out, 3, self.channels_axis)
        return Pade32a_(w_scale=w_scale, w_bias=w_bias, w_a=w_a)


# =============================================================================
# Not exported (see __all__): kept private, not deleted, in case a future
# use case wants the build-then-apply construction below instead of the
# inline `RQSplineContextModule_` in `rqs_modules_.py`.
class _RQSplineContextModule_(ContextModule_):
    """
    A wrapper for `RQSplineNet_` that uses a rational quadratic spline (RQS)
    to transform distributions, with its knots computed from a context
    tensor via `feature_map_fn`.

    Basically identical to `RQSplineContextModule_` in `rqs_modules_.py`:
    both compute knots from `feature_map_fn(*args)` via `make_rq_spline_field`.
    The difference is architectural, not mathematical -- this class builds the
    parameterized spline via `make_conditioned_transform` (inherited dispatch
    from `ContextModule_`) instead of applying it inline.

    At instantiation, this module accepts a callable (typically an instance
    of `torch.nn.Module`), which is responsible for transforming the context
    tensor(s) into the features used to build the spline. The knots are derived
    from that output by `make_rq_spline_field` (see `lib.spline` for the exact
    convention and the full set of options, e.g. `smooth`).

    Parameters
    ----------
    feature_map_fn : Callable
        Function (or `torch.nn.Module`) producing, from the context, the
        features that parameterize the spline.
    channels_axis : int, optional
        Axis of `feature_map_fn`'s output interpreted as the knots axis.
        Default: -1.
    xlim, ylim : tuple of float, optional
        Minimum and maximum values for x and y. Defaults to (0, 1).
    knots_x, knots_y : Tensor or None, optional
        If provided, these fix the knot positions instead of learning them.
    smooth : bool, optional
        If True, enforce smooth derivatives across knots. Default: False.
    extrap : dict or None, optional
        Extrapolation behavior outside the domain.
    """

    def __init__(
        self,
        feature_map_fn: Callable,
        channels_axis: int = -1,
        xlim: Tuple[float, float] = (0, 1),
        ylim: Tuple[float, float] = (0, 1),
        knots_x: Tensor | None = None,
        knots_y: Tensor | None = None,
        smooth: bool = False,
        extrap: Dict | None = None
    ):
        super().__init__()

        self.feature_map_fn = feature_map_fn
        self.channels_axis = channels_axis
        self.xlim = xlim
        self.ylim = ylim
        self.knots_x = knots_x
        self.knots_y = knots_y
        self.smooth = smooth
        self.extrap = extrap or {}

    def make_conditioned_transform(self, args=None) -> Module_:
        """
        Build an `ApplySpline_` whose knots come from:

            out = feature_map_fn(args)

        The resulting feature tensor is handed to `make_rq_spline_field`.
        (see `lib.spline` for the exact convention used to derive knot
        positions and derivatives).

        If `out` has a dummy leading batch dimension of size 1 (e.g. from a
        `feature_map_fn` that does not depend on a real per-sample batch),
        it is squeezed off before computing the knots.
        """
        out = self.feature_map_fn(*self._as_args_tuple(args))
        if out.shape[0] == 1:
            out = out.squeeze(0)  # squeeze the dummy batch dim

        spline = make_rq_spline_field(
            out,
            xlim=self.xlim,
            ylim=self.ylim,
            knots_axis=self.channels_axis,
            knots_x=self.knots_x,
            knots_y=self.knots_y,
            smooth=self.smooth,
            extrap=self.extrap,
        )
        return ApplySpline_(spline)


# =============================================================================
class ApplySpline_(Module_):
    """
    Applies a pre-built spline object to `x`.

    Input `x` is handed to the spline as is, so it must have the same number
    of dimensions as the knots and agree with them on every axis but the knots
    axis. For knots built from a feature map of shape `(*shape, n_features)`,
    i.e. knots of shape `(*shape, n_knots)`, this means `x` of shape
    `(*shape, k)` for any `k`. Reshaping `x` into that form is the caller's
    responsibility.
    """
    def __init__(self, spline):
        super().__init__()
        self.spline = spline

    def forward(self, x, log0=0):
        fx, g = self.spline(x, grad=True)  # g is gradient @ x
        logj = self.sum_density(torch.log(g))
        return fx, log0 + logj

    def reverse(self, x, log0=0):
        fx, g = self.spline.reverse(x, grad=True)  # g is gradient @ x
        logj = self.sum_density(torch.log(g))
        return fx, log0 + logj
