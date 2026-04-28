# Copyright (c) 2021-2025 Javad Komijani

"""
This module includes several basic subclasses of `Module_` that are designed
specifically for scalar tensors. These subclasses implement various common
transformations, making them useful for applications in machine learning,
particularly in probabilistic modeling and generative tasks.

The trailing underscore in `Module_` and its subclasses indicates that the
`forward` and `reverse` methods return a two-item tuple:
   1. The transformed input.
   2. The logarithm of the Jacobian determinant of the transformation,
      which is useful in contexts like normalizing flows or transformations
      that involve volume changes.
"""

# pylint: disable=relative-beyond-top-level
# pylint: disable=too-many-arguments, too-many-positional-arguments
# pylint: disable=invalid-name


from typing import Union

import torch
import numpy as np

from .modules import SplineNet
from .._core import Module_, ModuleList_


__all__ = [
    "Identity_",
    "Clone_",
    "SoftSqrt_",
    "Affine_",
    "Pade11_",
    "Pade22_",
    "Pade32_",
    "Pade32a_",
    "Pade22Spline_",  # same as UnityDistConvertor_
    "DistConvertor_",
    "UnityDistConvertor_",
    "PhaseDistConvertor_"
]

Number = Union[int, float, complex]
Tensor = torch.Tensor


# =============================================================================
class Identity_(Module_):
    """
    Identity transformation.

    Applies the identity map to inputs: forward and reverse both
    return `(x, log0)` unchanged.
    """

    def forward(self, x, log0=0, **extra):
        return x, log0

    def reverse(self, x, log0=0, **extra):
        return x, log0


class Clone_(Module_):
    """
    Clone transformation.

    Forward and reverse both return a cloned copy of `x`, leaving
    `log0` unchanged.
    """

    def forward(self, x, log0=0, **extra):
        return x.clone(), log0

    def reverse(self, x, log0=0, **extra):
        return x.clone(), log0


class Tanh_(Module_):
    """
    Hyperbolic tangent transformation.

    Applies `tanh` in the forward direction, adjusting `log0` with
    the log-Jacobian. Reverse is implemented with `ArcTanh_`.
    """

    def forward(self, x, log0=0):
        logj = -2 * self.sum_density(torch.log(torch.cosh(x)))
        return torch.tanh(x), log0 + logj

    def reverse(self, x, log0=0):
        return ArcTanh_().forward(x, log0)


class ArcTanh_(Module_):
    """
    Inverse hyperbolic tangent transformation.

    Applies `atanh` in the forward direction, adjusting `log0` with
    the log-Jacobian. Reverse is implemented with `Tanh_`.
    """

    def forward(self, x, log0=0):
        y = torch.atanh(x)
        logj = 2 * self.sum_density(torch.log(torch.cosh(y)))
        return y, log0 + logj

    def reverse(self, x, log0=0):
        return Tanh_().forward(x, log0)


class Expit_(Module_):
    """
    Logistic sigmoid transformation.

    Also called `Sigmoid_`. Applies `expit` in the forward direction,
    with exact log-Jacobian. Reverse is implemented with `Logit_`.
    """

    def forward(self, x, log0=0):
        y = 1 / (1 + torch.exp(-x))
        logj = self.sum_density(-x + 2 * torch.log(y))
        return y, log0 + logj

    def reverse(self, x, log0=0):
        return Logit_().forward(x, log0)


class Logit_(Module_):
    """
    Logit transformation.

    Inverse of `Sigmoid_`. Maps (0, 1) to (-inf, inf) in the forward
    direction with exact log-Jacobian. Reverse is implemented with
    `Expit_`.
    """

    def forward(self, x, log0=0):
        y = torch.log(x / (1 - x))
        logj = -self.sum_density(torch.log(x * (1 - x)))
        return y, log0 + logj

    def reverse(self, x, log0=0):
        return Expit_().forward(x, log0)


class SoftSqrt_(Module_):
    """
    Signed Soft Square-Root Transformation.

    This transformation smoothly maps real values `x` to `y` using a signed
    square-root operation with a small positive offset `eps` for numerical
    stability when computing the log-Jacobian.

    Forward transformation:
        y = sign(x) * sqrt(eps^2 + |x|)

    Inverse (reverse) transformation:
        x = sign(y) * (|y|^2 - eps^2)
    """

    def __init__(self, eps=1e-4):
        super().__init__()
        # Store squared epsilon to avoid recomputation
        self.eps_sq = eps ** 2

    def forward(self, x: torch.Tensor, log0=0):
        """Apply the forward signed soft square-root transformation."""
        y = torch.sign(x) * torch.sqrt(self.eps_sq + torch.abs(x))
        logj = -self.sum_density(torch.log(torch.abs(y) * 2))
        return y, log0 + logj

    def reverse(self, y: torch.Tensor, log0=0):
        """Apply the reverse transformation."""
        x = torch.sign(y) * (torch.abs(y) ** 2 - self.eps_sq)
        logj = -self.sum_density(torch.log(torch.abs(y) * 2))
        return x, log0 - logj


# =============================================================================
# The following Modules may have trainable parameters
# =============================================================================
class Affine_(Module_):
    """
    An affine transformation, :math:`a x + b`, with trainable parameters.

    This module treats the parameters :math:`a, b` as trainable by default.
    To ensure the invertibility of the transformation, the `Softplus` function
    is utilized to impose the constraint :math:`a > 0`.
    Furthermore, if the input data includes a channel axis, it is possible to
    specify different parameters for each channel, thereby accommodating a
    variety of use cases.

    Parameters
    ----------
    channels_axis : Union[int, None], optional
        Indicates the axis corresponding to the channels in the input data.
        The default value is None, which means that there are no channels
        in the input.

    n_channels : int, optional
        Specifies the number of channels when `channels_axis` is set to an
        integer value. This parameter becomes irrelevant if `channels_axis`
        is None.

    w_scale: Union[Tensor, Number, None], optional
        The default is None, meaning :math:`a` is a trainable parameter. If
        provided, :math:`a` is set to `Softplus(w_{scale}, log(2))`.

    w_bias: Union[Tensor, Number, None], optional
        The default is None, meaning :math:`b` is a trainable parameter. If
        provided, :math:`b` is set to `w_{bias}`.
    """

    softplus = torch.nn.Softplus(beta=np.log(2))
    # with beta = log(2), we have softplust(0) = 1

    def __init__(
        self,
        channels_axis: Union[int, None] = None,
        n_channels: int = 1,
        w_scale: Union[Tensor, Number, None] = None,
        w_bias: Union[Tensor, Number, None] = None
    ):

        super().__init__()

        if channels_axis is None:
            assert n_channels == 1

        if w_scale is None:
            w_scale = torch.nn.Parameter(torch.zeros(n_channels))

        if w_bias is None:
            w_bias = torch.nn.Parameter(torch.zeros(n_channels))

        self.w_scale = w_scale
        self.w_bias = w_bias
        self.n_channels = n_channels
        self.channels_axis = channels_axis

    def forward(self, x, log0=0):
        scale, bias = self.get_parameters_reshaped(x.shape)
        logj = self.sum_density(torch.log(scale) * torch.ones_like(x))
        return scale * x + bias, log0 + logj

    def reverse(self, y, log0=0):
        scale, bias = self.get_parameters_reshaped(y.shape)
        logj = - self.sum_density(torch.log(scale) * torch.ones_like(y))
        return (y - bias) / scale, log0 + logj

    def get_parameters_reshaped(self, shape):
        if self.channels_axis is None:
            w_scale = self.w_scale
            w_bias = self.w_bias
        else:
            shape = [1 for _ in shape]
            shape[self.channels_axis] = self.n_channels
            w_scale = self.w_scale.reshape(*shape)
            w_bias = self.w_bias.reshape(*shape)
        return self.softplus(w_scale), w_bias


class Pade11_(Module_):
    r"""An invertible transformation as a Pade approximant of order [1/1],

    .. math::

        f(x; a) = \frac{x}{x + a (1 - x)}

    where :math:`a > 0`. The function maps the interval :math:`[0, 1]` to
    itself, making it particularly useful for modeling input and output
    variables that are confined within the range of zero and one.

    This module treats the parameter :math:`a` as trainable by default.
    To ensure the invertibility of the transformation, the `Softplus` function
    is utilized to impose the necessary constraints.
    Furthermore, if the input data includes a channel axis, it is possible to
    specify different parameters for each channel, thereby accommodating a
    variety of use cases.

    Note that this transformation can also be expressed in an equivalent
    mathematical form as `expit(logit(x) - log(a))`. Here, the function `expit`
    is also known as the logistic sigmoid function, and `logit` is its inverse.
    Furthermore, the inverse of this transformation is defined as
    :math:`f(y; 1/a)`.

    Parameters
    ----------
    channels_axis : Union[int, None], optional
        Indicates the axis corresponding to the channels in the input data.
        The default value is None, which means that there are no channels
        in the input.

    n_channels : int, optional
        Specifies the number of channels when `channels_axis` is set to an
        integer value. This parameter becomes irrelevant if `channels_axis`
        is None.
    """

    softplus = torch.nn.Softplus(beta=np.log(2))
    # with beta = log(2), we have softplust(0) = 1

    def __init__(
        self,
        channels_axis: Union[int, None] = None,
        n_channels: int = 1
    ):

        super().__init__()

        if channels_axis is None:
            assert n_channels == 1

        self.w1 = torch.nn.Parameter(torch.zeros(n_channels))
        self.n_channels = n_channels
        self.channels_axis = channels_axis

    def forward(self, x, log0=0):
        d1 = self.get_parameters_reshaped(x.shape)
        denom = x + (1 - x) * d1
        logj = self.sum_density(torch.log(d1) - 2 * torch.log(denom))
        return x / denom, log0 + logj

    def reverse(self, y, log0=0):
        d1 = self.get_parameters_reshaped(y.shape)
        denom = y + (1 - y) / d1
        logj = self.sum_density(-torch.log(d1) - 2 * torch.log(denom))
        return y / denom, log0 + logj

    def get_parameters_reshaped(self, shape):
        if self.channels_axis is None:
            w1 = self.w1
        else:
            shape = [1 for _ in shape]
            shape[self.channels_axis] = self.n_channels
            w1 = self.w1.reshape(*shape)
        return self.softplus(w1)


class Pade22_(Module_):
    r"""An invertible transformation as a Pade approximant of order [2/2],

    .. math::

        f(x; a, b) = \frac{x^2 + a x (1 - x)}{1 + b x (1 - x)}

    where :math:`a > 0` and :math:`b > a - 2`. The function maps the interval
    :math:`[0, 1]` to itself, making it particularly useful for modeling input
    and output variables that are confined within the range of zero and one.

    This module treats the parameter :math:`a` as trainable by default.
    To ensure the invertibility of the transformation, the `Softplus` function
    is utilized to impose the necessary constraints.
    Furthermore, if the input data includes a channel axis, it is possible to
    specify different parameters for each channel, thereby accommodating a
    variety of use cases.

    Parameters
    ----------
    channels_axis : Union[int, None], optional
        Indicates the axis corresponding to the channels in the input data.
        The default value is None, which means that there are no channels
        in the input.

    n_channels : int, optional
        Specifies the number of channels when `channels_axis` is set to an
        integer value. This parameter becomes irrelevant if `channels_axis`
        is None.

    symmetric : bool, optional
        Determines whether the transformation is symmetric with respect to
        the point `[0.5, 0.5]`. The default value is False, indicating this
        symmetry is not imposed by construction.
    """

    softplus = torch.nn.Softplus(beta=np.log(2))
    # with beta = log(2), we have softplust(0) = 1

    def __init__(
        self,
        channels_axis: Union[int, None] = None,
        n_channels: int = 1,
        symmetric: bool = False
    ):

        super().__init__()

        if channels_axis is None:
            assert n_channels == 1

        self.w0 = torch.nn.Parameter(torch.zeros(n_channels))
        if not symmetric:
            self.w1 = torch.nn.Parameter(torch.zeros(n_channels))
        else:
            self.w1 = self.w0

        self.n_channels = n_channels
        self.channels_axis = channels_axis
        self.symmetric = symmetric

    def forward(self, x, log0=0):
        d0, d1 = self.get_parameters_reshaped(x.shape)
        denom = (1 + (d1 + d0 - 2) * x * (1 - x))
        g_0 = x * (x + d0 * (1 - x)) / denom
        g_1 = (d0 + 2 * (1 - d0) * x + (d1 + d0 - 2) * x**2) / denom**2
        return g_0, log0 + self.sum_density(torch.log(g_1))

    def reverse(self, y, log0=0):
        d0, d1 = self.get_parameters_reshaped(y.shape)
        x = self.reverse_pade22(y, d0, d1)
        denom = (1 + (d1 + d0 - 2) * x * (1 - x))
        g_1 = (d0 + 2 * (1 - d0) * x + (d1 + d0 - 2) * x**2) / denom**2
        return x, log0 - self.sum_density(torch.log(g_1))

    def get_parameters_reshaped(self, shape):
        if self.channels_axis is None:
            w0 = self.w0
            w1 = self.w1
        else:
            shape = [1 for _ in shape]
            shape[self.channels_axis] = self.n_channels
            w0 = self.w0.reshape(*shape)
            w1 = self.w1.reshape(*shape)

        return self.softplus(w0), self.softplus(w1)

    @staticmethod
    def reverse_pade22(y, d0, d1):
        r"""Return the solution of :math:`a x^2 + b x + c = 0,  x \in [0, 1]`,
        where the coefficients correspond to Pade [2, 2] map.

        Using the facts about :math:`x, y, d_0, and d_1`, one can show that the
        positive solution of the quadratic equation is

        .. math::

            x = (-b - \delta) / (2 * a)

        Because the expression is not well-defined for a vanishing `a`, we use
        the following identical expression

        .. math::

            x = 2 c / (-b + \delta)
        """
        b = (d1 + d0 - 2) * y - d0
        delta = torch.sqrt(b**2 + 4 * y * (1 + b))
        x = 2 * y / (delta - b)
        return x


class Pade32_(Module_):
    r"""
    An invertible transformation as a Pade approximant of order [3/2],

    .. math::

        f(x) = a x \frac{a + x^2}{1 + a x^2}

    which is monotonically increasing for all real values of :math:`x`,
    provided that :math:`0 < a < 3`.

    This module treats the parameter :math:`a` as trainable by default.
    To ensure the invertibility of the transformation, the `Expit` function
    is utilized to impose the necessary constraint.
    Furthermore, if the input data includes a channel axis, it is possible to
    specify different parameters for each channel, thereby accommodating a
    variety of use cases.

    Note that this transformation is not the most general invertible Pade
    [3/2], but it has the following traits: it is odd and analytic on the
    real axis, and asymptotic to the identity transformation for large
    values of :math:`|x|`.

    The inversion is possible by solving a cubic equation, which has only one
    real solution.

    Parameters
    ----------
    channels_axis : Union[int, None], optional
        Indicates the axis corresponding to the channels in the input data.
        The default value is None, which means that there are no channels
        in the input.

    n_channels : int, optional
        Specifies the number of channels when `channels_axis` is set to an
        integer value. This parameter becomes irrelevant if `channels_axis`
        is None.

    w_a: Union[Tensor, Number, None], optional
        The default is None, indicating that :math:`a` is a trainable
        parameter. If provided, :math:`a` is set to `3 expit(w_a - log(2))`.
    """

    def __init__(
        self,
        channels_axis: Union[int, None] = None,
        n_channels: int = 1,
        w_a: Union[Tensor, Number, None] = None
    ):

        super().__init__()

        if channels_axis is None:
            assert n_channels == 1

        if w_a is None:
            # We introduce parameter w_a, and then `a = 3 expit(w_a - log(2))`.
            # Note that 3 expit(-log(2)) = 1, indicating no nonlinearity
            w_a = torch.nn.Parameter(torch.randn(n_channels))

        self.w_a = w_a
        self.channels_axis = channels_axis
        self.n_channels = n_channels

    def forward(self, x, log0=0):
        a = self.get_parameters_reshaped(x.shape)  # a is derivative at x = 0
        s = x**2
        y = a * x * (a + s) / (1 + a * s)
        dy_by_dx = a * (a * s**2 + (3 - a**2) * s + a) / (1 + a * s)**2
        logj = self.sum_density(torch.log(dy_by_dx))
        return y, log0 + logj

    def reverse(self, y, log0=0):
        a = self.get_parameters_reshaped(y.shape)  # a is derivative at x = 0
        x = self.reverse_pade32(y / a, a)
        s = x**2
        dy_by_dx = a * (a * s**2 + (3 - a**2) * s + a) / (1 + a * s)**2
        logj = - self.sum_density(torch.log(dy_by_dx))
        return x, log0 + logj

    def get_parameters_reshaped(self, shape):
        if self.channels_axis is None:
            w_a = self.w_a
        else:
            shape = [1 for _ in shape]
            shape[self.channels_axis] = self.n_channels
            w_a = self.w_a.reshape(*shape)
        return 3 * torch.special.expit(w_a - np.log(2))

    @staticmethod
    def reverse_pade32(y, a):
        """
        Invert the rational function

            f(x) = x (a + x^2) / (1 + a x^2)

        where 0 < a < 3, by solving the equivalent cubic equation for x.
        This function computes the unique real solution for x given `y = f(x)`.

        Parameters
        ----------
        y : torch.Tensor or float
            The value of the function f(x) to invert.
        a : float
            Parameter of the rational function, must satisfy 0 < a < 3.

        Returns
        -------
        x : torch.Tensor or float
            The unique real solution of f(x) = y.

        Notes
        -----
        - f(x)/x ≥ 0 for all x ≠ 0, with f(0) = 0.
        - The inversion reduces to solving a cubic equation with one real root.
        - To ensure numerical stability, we introduce sgn(y) explicitly.
          A previous version avoided this but required special handling for
          y = 0.
        - Here, we use a sign-adjusted formulation that handles all cases
          uniformly.
        """

        # Compute discriminant-like terms (del0, del1) adapted for stability
        # Then, adjust del1 by sign(y)
        del0 = (a * y)**2 - 3 * a
        del1 = -2 * (a * y)**3 + (9 * a**2 - 27) * y
        del1 *= torch.sgn(y)

        # Compute the cubic solution via Cardano's method
        delta = 2**(-1/3) * (-del1 + torch.sqrt(del1**2 - 4 * del0**3))**(1/3)

        # Final solution: sign-adjusted root ensuring the correct branch
        x = (a * y + (delta + del0 / delta) * torch.sgn(y)) / 3

        return x


class Pade32a_(ModuleList_):
    r"""
    An invertible transformation based on a [3/2] Pade approximant, implemented
    through sequential instances of `Affine_` and `Pade32_`.

    The transformation is given by:

    .. math::

        f(x) = a z \frac{a + z^2}{1 + a z^2}

    where :math:`z = s x + b`. This function is invertible and monotonically
    increasing for all real values of :math:`x`, provided that
    :math:`0 < a < 3` and :math:`s > 0`.

    For further details of the transformation see `Affine_` and `Pade32_`.

    This module treats the parameters :math:`s, b, a` as trainable by default.
    To ensure the invertibility of the transformation, the `Softplus` and
    `Expit` functions are utilized to impose the necessary constraints.
    Furthermore, if the input data includes a channel axis, it is possible to
    specify different parameters for each channel, thereby accommodating a
    variety of use cases.

    Parameters
    ----------
    channels_axis : Union[int, None], optional
        Indicates the axis corresponding to the channels in the input data.
        The default value is None, which means that there are no channels
        in the input.

    n_channels : int, optional
        Specifies the number of channels when `channels_axis` is set to an
        integer value. This parameter becomes irrelevant if `channels_axis`
        is None.

    w_scale: Union[Tensor, Number, None], optional
        The default is None, meaning :math:`s` is a trainable parameter. If
        provided, :math:`s` is set to `Softplus(w_{scale}, log(2))`.

    w_bias: Union[Tensor, Number, None], optional
        The default is None, meaning :math:`b` is a trainable parameter. If
        provided, :math:`b` is set to `w_{bias}`.

    w_a: Union[Tensor, Number, None], optional
        The default is None, indicating that :math:`a` is a trainable
        parameter. If provided, :math:`a` is set to `3 expit(w_a - log(2))`.
    """

    def __init__(
        self,
        channels_axis: Union[int, None] = None,
        n_channels: int = 1,
        w_scale: Union[Tensor, Number, None] = None,
        w_bias: Union[Tensor, Number, None] = None,
        w_a: Union[Tensor, Number, None] = None
    ):

        affine_ = Affine_(channels_axis, n_channels, w_scale, w_bias)
        pade32_ = Pade32_(channels_axis, n_channels, w_a)

        super().__init__([affine_, pade32_])

    def reset_weights(self, w_scale, w_bias, w_a):
        self[0].w_scale = w_scale
        self[0].w_bias = w_bias
        self[1].w_a = w_a


class SplineNet_(SplineNet, Module_):
    """Identical to SplineNet, except for calculating the Jacobian of the
    transformation and returning its logarithm.

    This can be used as a probability distribution convertor for random
    variables with nonzero probability in [0, 1].
    """

    def forward(self, x, log0=0):
        spline = self.make_spline()
        x_reshaped = x.reshape(*self.spline_shape, -1)
        fx, g = spline(x_reshaped, grad=True)  # g is gradient @ x
        fx, g = fx.reshape(x.shape), g.reshape(x.shape)
        logj = self.sum_density(torch.log(g))
        return fx, log0 + logj

    def reverse(self, x, log0=0):
        spline = self.make_spline()
        x_reshaped = x.reshape(*self.spline_shape, -1)
        fx, g = spline.reverse(x_reshaped, grad=True)  # g is gradient @ x
        fx, g = fx.reshape(x.shape), g.reshape(x.shape)
        logj = self.sum_density(torch.log(g))
        return fx, log0 + logj


class UnityDistConvertor_(SplineNet_):
    """As a PDF convertor for random variables in range [0, 1]."""

    def __init__(self, knots_len, symmetric=False, **kwargs):

        if symmetric:
            extra = dict(xlim=(0.5, 1), ylim=(0.5, 1), extrap={'left': 'anti'})
        else:
            extra = {}

        super().__init__(knots_len, **kwargs, **extra)


class PhaseDistConvertor_(SplineNet_):
    """As a PDF convertor for random variables in range [-pi, pi]."""

    def __init__(self, knots_len, symmetric=False, **kwargs):

        pi = np.pi

        if symmetric:
            extra = dict(xlim=(0, pi), ylim=(0, pi), extrap={'left': 'anti'})
        else:
            extra = dict(xlim=(-pi, pi), ylim=(-pi, pi))

        super().__init__(knots_len, **kwargs, **extra)


class DistConvertor_(ModuleList_):
    """As a PDF convertor for real random variables.

    Steps: pass through instances of `Expit_`, `SplineNet_`, and `Logit_`.

    If `final_scale` is True, an instance of `Affine_` without a bias term is
    also included.
    """

    def __init__(
            self, knots_len, symmetric=False, final_scale=False, **kwargs
            ):

        assert knots_len > 1, f"SplineNet is not defined for {knots_len} knots"

        if symmetric:
            extra = dict(xlim=(0.5, 1), ylim=(0.5, 1), extrap={'left': 'anti'})
        else:
            extra = dict(xlim=(0, 1), ylim=(0, 1))

        nets_ = [Expit_(), SplineNet_(knots_len, **kwargs, **extra), Logit_()]

        if final_scale:
            nets_.append(Affine_(w_bias=0))

        super().__init__(nets_)

    @property
    def spline_layer_(self):
        return self[1]


class SgnBiasNet_(Module_):
    """This module should be used only and only in the first layer, where the
    input does not depend on the parameters of the net. Otherwise, because it
    is not continuous, the derivatives will be messed up.
    """

    def __init__(self, size=(1,)):
        super().__init__()
        self.w = torch.nn.Parameter(torch.rand(*size)/10)

    def forward(self, x, log0=0):
        return x + torch.sgn(x) * self.w**2, log0

    def reverse(self, x, log0=0):
        return x - torch.sgn(x) * self.w**2, log0


Pade22Spline_ = UnityDistConvertor_  # alias
