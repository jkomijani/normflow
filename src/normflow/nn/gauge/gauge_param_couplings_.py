# Copyright (c) 2021-2023 Javad Komijani

"""
This module contains new neural networks that are subclasses of `Module_`
and couple sites to each other.
"""

# pylint: disable=relative-beyond-top-level, arguments-differ, too-many-locals
# pylint: disable=too-many-arguments, too-few-public-methods
# pylint: disable=invalid-name

import torch
import numpy as np

from ..scalar.modules_ import Logit_, Expit_
from ..scalar.couplings_ import AffineCoupling_, Coupling_
from ..scalar.couplings_ import RQSplineCoupling_, MultiRQSplineCoupling_


# =============================================================================
class Pade11Coupling_(Coupling_):
    r"""An invertible transformation as a Pade approximant of order 1/1

    .. math::

        f(x; a) = x / (x + a * (1 - x))

    with :math:`a > 0` that maps :math:`[0, 1] \to [0, 1]`. This map is useful
    for input and output variables that vary between zero and one.

    This transformation is equivalent to math:`\expit(\logit(x) - \log(a))`
    and its inverse is :math:`f(y; 1/a)`.
    """

    softplus = torch.nn.Softplus(beta=np.log(2))

    def atomic_forward(self, *, x_active, x_frozen, parity, net, log0=0):
        """Forward pass of a single coupling layer."""
        t = net(x_frozen)
        t = self.mask.purify(t, channel=parity)
        d1 = self.softplus(t)

        def pade11_(x):
            y = x / (x + (1 - x) * d1)
            J = d1 / (x + (1 - x) * d1)**2
            return y, self.sum_density(torch.log(J))

        x_active, logJ = pade11_(x_active)

        return x_active, log0 + logJ

    def atomic_reverse(self, *, x_active, x_frozen, parity, net, log0=0):
        """Reverse pass of a single coupling layer."""
        t = net(x_frozen)
        t = self.mask.purify(t, channel=parity)
        d1 = self.softplus(t)

        def invpade11_(y):
            x = y / (y + (1 - y) / d1)
            J = 1 / d1 / (y + (1 - y) / d1)**2
            return x, self.sum_density(torch.log(J))

        x_active, logJ = invpade11_(x_active)

        return x_active, log0 + logJ


# =============================================================================
class Pade22Coupling_(Coupling_):
    r"""An invertible transformation as a Pade approximant of order 2/2

    .. math::

        f(x; a, b) = (x^2 + a x (1 - x)) / (1 + b x (1 - x))

    with :math:`a, b > 0` that maps :math:`[0, 1] \to [0, 1]`. This map is
    useful for input and output variables that vary between zero and one.
    """

    softplus = torch.nn.Softplus(beta=np.log(2))

    def atomic_forward(self, *, x_active, x_frozen, parity, net, log0=0):
        """Forward pass of a single coupling layer."""
        t = net(x_frozen)
        t = self.mask.purify(t, channel=parity)
        d0, d1 = self.softplus(t).chunk(2, dim=self.channels_axis)

        def pade22_(x):
            denom = (1 + (d1 + d0 - 2) * x * (1 - x))
            y = x * (x + d0 * (1 - x)) / denom
            J = (d0 + 2 * (1 - d0) * x + (d1 + d0 - 2) * x**2) / denom**2
            return y, self.sum_density(torch.log(J))

        x_active, logJ = pade22_(x_active)

        return x_active, log0 + logJ

    def atomic_reverse(self, *, x_active, x_frozen, parity, net, log0=0):
        """Reverse pass of a single coupling layer."""
        t = net(x_frozen)
        t = self.mask.purify(t, channel=parity)
        d0, d1 = self.softplus(t).chunk(2, dim=self.channels_axis)

        def invpade22_(y):
            x = self.reverse_pade22(y, d0, d1)
            denom = (1 + (d1 + d0 - 2) * x * (1 - x))
            inv_J = (d0 + 2 * (1 - d0) * x + (d1 + d0 - 2) * x**2) / denom**2
            return x, - self.sum_density(torch.log(inv_J))

        x_active, logJ = invpade22_(x_active)

        return x_active, log0 + logJ

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
        c = y
        b = (d1 + d0 - 2) * y - d0
        # a = -1 - b  # no need to define `a` (it is already plugged in below).
        delta = torch.sqrt(b**2 + 4 * c * (1 + b))
        # x = (-b - delta) / (2 * a)
        # x[a == 0] = (-c / b)[a == 0]
        x = 2 * c / (-b + delta)
        return x


# =============================================================================
class SU3RQSplineCoupling_(MultiRQSplineCoupling_):
    """Identical to the default of MultiRQSplineCoupling_."""


# =============================================================================
class SU2RQSplineCoupling_(RQSplineCoupling_):
    """Like RQSplineCoupling_, but with existing channel axis in input."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, handle_channel_axis=False, **kwargs)


# =============================================================================
class U1RQSplineCoupling_(SU2RQSplineCoupling_):
    """Like `SU2RQSplineCoupling_`."""

    # Ideally the following preprocessing should be done in embeded in "net"
    # def preprocess_fz(self, x):  # fz: frozen
    #    x = (2 * np.pi) * x
    #    return torch.cat((torch.cos(x), torch.sin(x)), dim=self.channels_axis)


# =============================================================================
class SUnParamAffineCoupling_(AffineCoupling_):
    """Like `AffineCoupling_`, but for bounded input/output."""

    logit_ = Logit_()
    expit_ = Expit_()

    def forward(self, x, log0=0):
        """Forward pass for maping the interval :math:`[0, 1]` to itself."""
        return self.expit_(*super().forward(*self.logit_(x, log0=log0)))

    def reverse(self, x, log0=0):
        """Reverse pass for maping the interval :math:`[0, 1]` to itself."""
        return self.expit_(*super().reverse(*self.logit_(x, log0=log0)))
