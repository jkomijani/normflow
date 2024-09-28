# Copyright (c) 2021-2024 Javad Komijani

"""
This module contains new neural networks that are subclasses of Module_ and
do not couple sites to each other.

As in Module_, the trailing underscore implies that the associated forward and
reverse methods handle the Jacobians of the transformation.
"""


import torch
import copy
import numpy as np
from typing import Union

from .modules import SplineNet
from .._core import Module_, ModuleList_


class Identity_(Module_):

    def __init__(self, label='identity_'):
        super().__init__(label=label)

    def forward(self, x, log0=0, **extra):
        return x, log0

    def reverse(self, x, log0=0, **extra):
        return x, log0


class Clone_(Module_):

    def __init__(self, label='clone_'):
        super().__init__(label=label)

    def forward(self, x, log0=0, **extra):
        return x.clone(), log0

    def reverse(self, x, log0=0, **extra):
        return x.clone(), log0


class ScaleNet_(Module_):
    """Scales the input by a positive weight."""
    # TODO: one can consider different weights for different channels

    softplus = torch.nn.Softplus(beta=np.log(2))

    def __init__(self, label='scale_'):
        super().__init__(label=label)
        self._weight = torch.nn.Parameter(torch.zeros(1))

    @property
    def weight(self):
        return self.softplus(self._weight)

    def forward(self, x, log0=0):
        return x * self.weight, log0 + self.log_jacobian(x.shape)

    def reverse(self, x, log0=0):
        return x / self.weight, log0 - self.log_jacobian(x.shape)

    def log_jacobian(self, x_shape):
        if Module_.propagate_density:
            return torch.log(self.weight) * torch.ones(x_shape)
        else:
            logwscaled = torch.log(self.weight) * np.prod(x_shape[1:])
            return logwscaled * torch.ones(x_shape[0], device=self._weight.device)


class Tanh_(Module_):

    def forward(self, x, log0=0):
        logJ = -2 * self.sum_density(torch.log(torch.cosh(x)))
        return torch.tanh(x), log0 + logJ

    def reverse(self, x, log0=0):
        return ArcTanh_().forward(x, log0)


class ArcTanh_(Module_):

    def forward(self, x, log0=0):
        y = torch.atanh(x)
        logJ = 2 * self.sum_density(torch.log(torch.cosh(y)))
        return y, log0 + logJ

    def reverse(self, x, log0=0):
        return Tanh_().forward(x, log0)


class Expit_(Module_):
    """This can be also called `Sigmoid_`."""

    def forward(self, x, log0=0):
        y = 1 / (1 + torch.exp(-x))
        logJ = self.sum_density(-x + 2 * torch.log(y))
        return y, log0 + logJ

    def reverse(self, x, log0=0):
        return Logit_().forward(x, log0)


class Logit_(Module_):
    """This is inverse of `Sigmoid_`."""

    def forward(self, x, log0=0):
        y = torch.log(x / (1 - x))
        logJ = - self.sum_density(torch.log(x * (1 - x)))
        return y, log0 + logJ

    def reverse(self, x, log0=0):
        return Expit_().forward(x, log0)


class Pade11_(Module_):
    r"""An invertible transformation as a Pade approximant of order [1/1],

    .. math::

        f(x; a) = x / (x + a * (1 - x))

    with :math:`a > 0` that maps :math:`[0, 1] \to [0, 1]`. This map is useful
    for input and output variables that vary between zero and one.

    This transformation is equivalent to math:`\expit(\logit(x) - \log(a))`
    and its inverse is :math:`f(y; 1/a)`.

    Parameters
    ----------
    channels_axis: Union[int, None], optional
        it specifies the axis corresponding to the channels in the input.
        Default is None, indicating there are no channels.

    n_channels: int, optional
        it specifies the number of channels if `channels_axis` is an integer;
        otherwise, it must be set 1, which is the default value.
    """

    softplus = torch.nn.Softplus(beta=np.log(2))

    def __init__(self,
                 channels_axis: Union[int, None] = None,
                 n_channels: int = 1
                 ):

        super().__init__()

        self.w1 = torch.nn.Parameter(torch.zeros(n_channels))
        self.n_channels = n_channels
        self.channels_axis = channels_axis

    def forward(self, x, log0=0):
        d1 = self.get_derivative_reshaped(x.shape)
        denom = x + (1 - x) * d1
        logJ = self.sum_density(torch.log(d1) - 2 * torch.log(denom))
        return x / denom, log0 + logJ

    def reverse(self, x, log0=0):
        d1 = self.get_derivative_reshaped(x.shape)
        denom = x + (1 - x) / d1
        logJ = self.sum_density(-torch.log(d1) - 2 * torch.log(denom))
        return x / denom, log0 + logJ

    def get_derivative_reshaped(self, shape):
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

        f(x; a, b) = (x^2 + a x (1 - x)) / (1 + b x (1 - x))

    with :math:`a, b > 0` that maps :math:`[0, 1] \to [0, 1]`. This map is
    useful for input and output variables that vary between zero and one.

    Parameters
    ----------
    channels_axis: Union[int, None], optional
        it specifies the axis corresponding to the channels in the input.
        Default is None, indicating there are no channels.

    n_channels: int, optional
        it specifies the number of channels if `channels_axis` is an integer;
        otherwise, it must be set 1, which is the default value.

    symmetric: bool, optional
        if True, the transformation is symmtetric wrt `[0.5, 0.5]`. Default is
        False.
    """

    softplus = torch.nn.Softplus(beta=np.log(2))

    def __init__(self,
                 channels_axis: Union[int, None] = None,
                 n_channels: int = 1,
                 symmetric: bool = False
                 ):

        super().__init__()

        self.w0 = torch.nn.Parameter(torch.zeros(n_channels))
        if not symmetric:
            self.w1 = torch.nn.Parameter(torch.zeros(n_channels))
        else:
            self.w1 = self.w0

        self.n_channels = n_channels
        self.channels_axis = channels_axis
        self.symmetric = symmetric

    def forward(self, x, log0=0):
        d0, d1 = self.get_derivatives_reshaped(x.shape)
        denom = (1 + (d1 + d0 - 2) * x * (1 - x))
        g_0 = x * (x + d0 * (1 - x)) / denom
        g_1 = (d0 + 2 * (1 - d0) * x + (d1 + d0 - 2) * x**2) / denom**2
        return g_0, log0 + self.sum_density(torch.log(g_1))

    def reverse(self, y, log0=0):
        d0, d1 = self.get_derivatives_reshaped(y.shape)
        x = self.reverse_pade22(y, d0, d1)
        denom = (1 + (d1 + d0 - 2) * x * (1 - x))
        g_1 = (d0 + 2 * (1 - d0) * x + (d1 + d0 - 2) * x**2) / denom**2
        return x, log0 - self.sum_density(torch.log(g_1))

    def get_derivatives_reshaped(self, shape):
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
        """Return the solution of :math:`a x^2 + b x + c = 0,  x \in [0, 1]`,
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
    r"""An invertible transformation as a Pade approximant of order [3/2],

    .. math::

        f(x) = x (a + x^2) / (1 + a x^2)

    which is invertible for :math:`0 < a < 3`. By default, this module treats
    :math:`a` as a trainable paramter, but there is an option to fix it to a
    given constant. Moreover, if the input has a channel axis, it is possible
    to consider different values of :math:`a` for each channel.

    Note that the above transformation is not the most general invertible
    Pade [3/2], but it has the following traits: it is odd and regular at
    any finite real :math:`x`, it has three fixed points at zero and plus/minus
    unity, and it is proportional to :math:`x` as :math:`x \to \pm \infty`.

    Furthere remarks:
    1.  For inversion, one should solve a cubic equation, which has only one
        real solution.
    1.  An interesting observation: :math:`f(1/x) = 1 / f(x)`.
    2.  The transformation is identity when :math:`a = 1`.
    3.  It can be used as a nonlinear activation (if :math:`a \neq 1`).

    Parameters
    ----------
    channels_axis: Union[int, None], optional
        it specifies the axis corresponding to the channels in the input.
        Default is None, indicating there are no channels.

    n_channels: int, optional
        it specifies the number of channels if `channels_axis` is an integer;
        otherwise, it must be set 1, which is the default value.

    w_0: Union[float, None], optional
        it is by default None, indicating that :math:`a` is considered
        a trainable paramter. Otherwise, we have :math:`a = 3 \expit(w_0)`.
    """

    def __init__(self,
                 channels_axis: Union[int, None] = None,
                 n_channels: int = 1,
                 w_0: Union[float, None] = None
                ):

        super().__init__()

        if w_0 is None:
            # We introduce parameter `w_0`, and then: `a = 3 expit(w_0)`.
            # The initial value for `w_0` is normal with mean `log(2)`.
            # Note that 3 expit(-log(2)) = 1, indicating no nonlinearity
            w_0 = torch.nn.Parameter(- np.log(2) + torch.randn(n_channels))

        self.w_0 = w_0
        self.channels_axis = channels_axis
        self.n_channels = n_channels

    def forward(self, x, log0=0):
        a = self.get_derivative_reshaped(x.shape)  # a is derivative at x = 0
        s = x**2
        y = x * (a + s) / (1 + a * s)
        dy_by_dx = (a * s**2 + (3 - a**2) * s + a) / (1 + a * s)**2
        logJ = self.sum_density(torch.log(dy_by_dx))
        return y, log0 + logJ

    def reverse(self, y, log0=0):
        a = self.get_derivative_reshaped(y.shape)  # a is derivative at x = 0
        x = self.reverse_pade32(y, a)
        s = x**2
        dy_by_dx = (a * s**2 + (3 - a**2) * s + a) / (1 + a * s)**2
        logJ = - self.sum_density(torch.log(dy_by_dx))
        return x, log0 + logJ

    def get_derivative_reshaped(self, shape):
        if self.channels_axis is None:
            w_0 = self.w_0
        else:
            shape = [1 for _ in shape]
            shape[self.channels_axis] = self.n_channels
            w_0 = self.w_0.reshape(*shape)
        return 3 * torch.special.expit(w_0)

    @staticmethod
    def reverse_pade32(y, a):
        """We solve a cubic relation that has only one real solution.

        More specfically, we would like to invert

        .. math::

            f(x) = x (a + x^2) / (1 + a x^2)

        where :math:`0 < a < 3`.
        """
        # `f(x) / x` is always positive unless for `x = 0`, where f(0) = 0`.
        del0 = a**2 - 3 * a / y**2
        del1 = - 2 * a**3 + (9 * a**2 - 27) / y**2
        delta = 2**(-1/3) * (- del1 + torch.sqrt(del1**2 - 4*del0**3))**(1/3)
        x = y * (a + delta + del0 / delta) / 3
        # The above algorithm works for all `y` but `y = 0`. For this special
        # case we use `torch.nan_to_num` to set to 0.
        x = torch.nan_to_num(x, nan=0., posinf=0., neginf=0.)
        return x


class SplineNet_(SplineNet, Module_):
    """Identical to SplineNet, except for taking care of log_jacobian.

    This can be used as a probability distribution convertor for random
    variables with nonzero probability in [0, 1].
    """

    def forward(self, x, log0=0):
        spline = self.make_spline()
        x_reshaped = x.reshape(*self.spline_shape, -1)
        fx, g = spline(x_reshaped, grad=True)  # g is gradient @ x
        fx, g = fx.reshape(x.shape), g.reshape(x.shape)
        logJ = self.sum_density(torch.log(g))
        return fx, log0 + logJ

    def reverse(self, x, log0=0):
        spline = self.make_spline()
        x_reshaped = x.reshape(*self.spline_shape, -1)
        fx, g = spline.reverse(x_reshaped, grad=True)  # g is gradient @ x
        fx, g = fx.reshape(x.shape), g.reshape(x.shape)
        logJ = self.sum_density(torch.log(g))
        return fx, log0 + logJ


class UnityDistConvertor_(SplineNet_):
    """As a PDF convertor for random variables in range [0, 1]."""

    def __init__(self, knots_len, symmetric=False, **kwargs):

        if symmetric:
            extra = dict(xlim=(0.5, 1), ylim=(0.5, 1), extrap={'left':'anti'})
        else:
            extra = {}

        super().__init__(knots_len, **kwargs, **extra)


class PhaseDistConvertor_(SplineNet_):
    """As a PDF convertor for random variables in range [-pi, pi]."""

    def __init__(self, knots_len, symmetric=False, label='phase-dc_', **kwargs):

        pi = np.pi

        if symmetric:
            extra = dict(xlim=(0, pi), ylim=(0, pi), extrap={'left':'anti'})
        else:
            extra = dict(xlim=(-pi, pi), ylim=(-pi, pi))

        super().__init__(knots_len, label=label, **kwargs, **extra)


class DistConvertor_(ModuleList_):
    """As a PDF convertor for real random variables (from minus to plus
    infinity).

    Steps: pass through Expit_, SplineNet_, and Logit_
    """

    def __init__(self, knots_len, symmetric=False, label='dc_',
            sgnbias=False, initial_scale=False, final_scale=False,
            **kwargs
            ):

        if symmetric:
            extra = dict(xlim=(0.5, 1), ylim=(0.5, 1), extrap={'left':'anti'})
        else:
            extra = dict(xlim=(0, 1), ylim=(0, 1))

        if knots_len > 1:
            spline_ = SplineNet_(knots_len, label='spline_', **kwargs, **extra)
            nets_ = [Expit_(label='expit_'), spline_, Logit_(label='logit_')]
        else:
            nets_ = []

        if initial_scale:
            nets_ = [ScaleNet_(label='scale_')] + nets_
        elif final_scale:
            nets_ = nets_ + [ScaleNet_(label='scale_')]

        if sgnbias:  # SgnBiasNet_() **must** come first if exits
            nets_ = [SgnBiasNet_()] + nets_

        super().__init__(nets_)
        self.label = label

    @property
    def spline_layer_(self):
        for net_ in self:
            if net_.label == 'spline_':
                return net_

    @property
    def scale_layer_(self):
        for net_ in self:
            if net_.label == 'scale_':
                return net_

    @property
    def sgnbias_layer_(self):
        for net_ in self:
            if net_.label == 'sgnbias_':
                return net_


class SgnBiasNet_(Module_):
    """This module should be used only and only in the first layer, where the
    input does not depend on the parameters of the net. Otherwise, because it
    is not continuous, the derivatives will be messed up.
    """

    def __init__(self, size=[1], label='sgnbias_'):
        super().__init__(label=label)
        self.w = torch.nn.Parameter(torch.rand(*size)/10)

    def forward(self, x, log0=0):
        return x + torch.sgn(x) * self.w**2, log0

    def reverse(self, x, log0=0):
        return x - torch.sgn(x) * self.w**2, log0
