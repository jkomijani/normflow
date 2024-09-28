# Copyright (c) 2021-2024 Javad Komijani

"""This module contains new neural networks..."""


import torch
import copy
import numpy as np
from typing import Union

from ...lib.spline import RQSpline
from ...lib.linalg import neighbor_mean
from .convNd import Conv4d


class AvgNeighborPool(torch.nn.Module):
    """Return average of all neighbors"""

    def forward(self, x):
        return neighbor_mean(x, dim=range(1, x.ndim))


class Abs(torch.nn.Module):
    """Added for adding to the list of activations"""

    def forward(self, x):
        return torch.abs(x)


class Expit(torch.nn.Module):
    """This can be also called Sigmoid and is basically torch.nn.Sigmoid"""

    def forward(self, x):
        return torch.special.expit


class Logit(torch.nn.Module):
    """This is inverse of Sigmoid"""

    def forward(self, x):
        return torch.special.logit


ACTIVATIONS = torch.nn.ModuleDict(
                    [['tanh', torch.nn.Tanh()],
                     ['relu', torch.nn.ReLU()],
                     ['leaky_relu', torch.nn.LeakyReLU()],
                     ['softplus', torch.nn.Softplus()],
                     ['avg_neighbor_pool', AvgNeighborPool()],
                     ['abs', Abs()],
                     ['expit', Expit()],
                     ['logit', Logit()],
                     ['none', torch.nn.Identity()]
                    ]
              )


class PlusBias(torch.nn.Module):

    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return x + self.bias


class ConvAct(torch.nn.Sequential):
    """
    As an extension to torch.nn.Conv2d, this network is a sequence of
    convolutional layers with possible hidden layers and activations and other
    dimensions.

    Instantiating this class with the default optional variables is equivalent
    to instantiating torch.nn.Conv2d with following optional varaibles:
    padding = 'same' and padding_mode = 'circular'.

    As an option, one can provide a list/tuple for `hidden_sizes`. Then, one
    must also provide another list/tuple for activations using the option
    `acts`; the lenght of `acts` must be equal to the lenght of `hidden_sizes`
    plus 1 (for the output layer).
    There is also another option for pre-activation of the input: `pre_act`.

    The axes of the input and output tensors are treated as
    :math:`tensor(:, ch, ...)`, where `:` stands for the batch axis,
    `ch` for the channels axis, and `...` for the features axes.

    .. math::

        out(:, ch_o, ...) = bias(ch_o) +
                        \sum_{ch_i} weight(ch_o, ...) \star input(:, ch_i, ...)

    where :math:`\star` is n-dimensional cross-correlation operator acting on
    the features axes. The supported features dinensions are 1, 2, 3, and 4.
    Note that is is possible to change the channels axis from 1 to any other
    axis.

    Parameters
    ----------
    in_channels (int):
        Number of channels in the input data
    out_channels (int):
        Number of channels produced by the convolution
    kernel_size (int or tuple):
        Size of the convolving kernel
    conv_dim (int, optional):
        Dimension of the convolving kernel (default is 2)
    hidden_sizes (list/tuple of int, optional):
        Sizes of hidden layers (default is [])
    acts (list/tuple of str or None, optional):
        Activations after each layer (default is None)
    pre_act (str or None, optional):
        A possible activation layer before the rest (default is None)
    channels_axis (int, optional):
        Specifies the channels axis (default is 1)
    """

    Conv = {1: torch.nn.Conv1d,
            2: torch.nn.Conv2d,
            3: torch.nn.Conv3d,
            4: Conv4d
            }

    def __init__(self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            conv_dim: int = 2,
            hidden_sizes = [],
            acts = [None],
            pre_act = None,
            channels_axis: int = 1,
            **extra_kwargs  # all other kwargs to pass to torch.nn.Conv?d
            ):

        Conv = self.Conv[conv_dim]
        sizes = [in_channels, *hidden_sizes, out_channels]
        assert len(acts) == len(hidden_sizes) + 1

        conv_kwargs = dict(padding='same', padding_mode='circular')
        conv_kwargs.update(extra_kwargs)

        nets = [] if pre_act is None else [ACTIVATIONS[pre_act]]

        for i, act in enumerate(acts):
            nets.append(Conv(sizes[i], sizes[i+1], kernel_size, **conv_kwargs))
            if act is not None:
                nets.append(ACTIVATIONS[act])

        super().__init__(*nets)

        # save all inputs so that the can be used later for transfer learning
        conv_kwargs.update(
                dict(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, conv_dim=conv_dim,
                     hidden_sizes=hidden_sizes, acts=acts, pre_act=pre_act,
                     channels_axis=channels_axis
                     )
                )
        self.conv_kwargs = conv_kwargs

    def forward(self, x):
        channels_axis = self.conv_kwargs['channels_axis']
        if channels_axis == 1:
            return super().forward(x)
        else:
            x = torch.movedim(x, channels_axis, 1)
            x = super().forward(x)
            return torch.movedim(x, 1, channels_axis)

    def set_param2zero(self):
        # Do NOT use this unless for test, otherwise, the params do not change
        for net in self:
            for param in net.parameters():
                torch.nn.init.zeros_(param)

    def set_param2normal(self, mean=0.0, std=1.0):
        for net in self:
            for param in net.parameters():
                torch.nn.init.normal_(param, mean=mean, std=std)

    def _outdated_transfer(self, scale_factor=1, **extra):
        # Outdated: must be updated and ...
        """
        Returns a copy of the current module if scale_factor is 1.
        Otherwise, uses the input scale_factor to resize the kernel size.
        """
        if scale_factor == 1:
            return copy.deepcopy(self)
        else:
            pass  # change the kernel size as below

        ksize = self.conv_kwargs['kernel_size']  # original kernel size
        ksize = 1 + 2 * round((ksize - 1) * scale_factor/2)  # new kernel size

        conv_kwargs = dict(**self.conv_kwargs)
        conv_kwargs['kernel_size'] = ksize

        new_size = [ksize] * conv_kwargs['conv_dim']
        resize = lambda p: torch.nn.functional.interpolate(p, size=new_size)

        state_dict_conv = {key: resize(value)
                for key, value in self.net[::2].state_dict().items()
                }

        state_dict_acts = {key: value
                for key, value in self.net[1::2].state_dict().items()
                }

        state_dict = dict(**state_dict_conv, **state_dict_acts)

        new_net = self.__class__(**conv_kwargs)
        new_net.net.load_state_dict(state_dict)

        return new_net


class LinearAct(torch.nn.Sequential):
    """
    As an extension to torch.nn.Linear, this network is a sequence of linear
    layers with possible hidden layers and activations.

    As an option, one can provide a list/tuple for `hidden_sizes`. Then, one
    must also provide another list/tuple for activations using the option
    `acts`; the lenght of `acts` must be equal to the lenght of `hidden_sizes`
    plus 1 (for the output layer).
    There is also another option for pre-activation of the input: `pre_act`.

    The axes of the input and output tensors are treated as
    :math:`tensor(..., f)`, where `...` stands for any number of dimensions
    and `f` for the features axis.

    Parameters
    ----------
    in_features (int):
        Number of features in the input data
    out_features (int):
        Number of features in the output data
    hidden_sizes (list/tuple of int, optional):
        Sizes of hidden layers (default is [])
    acts (list/tuple of str or None, optional):
        Activations after each layer (default is None)
    pre_act (str or None, optional):
        A possible activation layer before the rest (default is None)
    """
    def __init__(self,
            in_features: int,
            out_features: int,
            hidden_sizes = [],
            acts = [None],
            pre_act = None,
            final_bias = False,  # e.g., can be used with 'abs' activation
            features_axis = -1,
            **linear_kwargs  # all other kwargs to pass to torch.nn.Linear
            ):

        Linear = torch.nn.Linear
        sizes = [in_features, *hidden_sizes, out_features]
        assert len(acts) == len(hidden_sizes) + 1

        nets = [] if pre_act is None else [ACTIVATIONS[pre_act]]

        for i, act in enumerate(acts):
            nets.append(Linear(sizes[i], sizes[i+1], **linear_kwargs))
            if act is not None:
                nets.append(ACTIVATIONS[act])

        if final_bias:
            nets.append(PlusBias(out_features))

        super().__init__(*nets)

        # save all inputs so that the can be used later for transfer learning
        linear_kwargs.update(
                dict(in_features=in_features, out_features=out_features,
                     hidden_sizes=hidden_sizes, acts=acts, pre_act=pre_act,
                     final_bias=final_bias, features_axis=features_axis
                     )
                )
        self.linear_kwargs = linear_kwargs

    def forward(self, x):
        features_axis = self.linear_kwargs['features_axis']
        if features_axis == -1:
            return super().forward(x)
        else:
            x = torch.movedim(x, features_axis, -1)
            x = super().forward(x)
            return torch.movedim(x, -1, features_axis)

    def set_param2zero(self):
        # Do NOT use this unless for test, otherwise, the params do not change
        for net in self:
            for param in net.parameters():
                torch.nn.init.zeros_(param)

    def set_param2normal(self, mean=0.0, std=1.0):
        for net in self:
            for param in net.parameters():
                torch.nn.init.normal_(param, mean=mean, std=std)


class Pade32(torch.nn.Module):
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
        y = x * (a + x**2) / (1 + a * x**2)
        return y

    def reverse(self, y, log0=0):
        a = self.get_derivative_reshaped(y.shape)  # a is derivative at x = 0
        x = self.reverse_pade32(y, a)
        return x

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


class SplineNet(torch.nn.Module):
    """
    Return a neural network for spline interpolation/extrapolation.
    The input `knots_len` specifies the number of knots of the spline.
    In general, the first knot is always at (xlim[0], ylim[0]) and the last
    knot is always at (xlim[1], ylim[1]) and the coordintes of other knots are
    network parameters to be trained, unless one explicitely provides
    `knots_x` and/or `knots_y`.
    Assuming `knots_x` is None, one needs `(knots_len - 1)` parameters to
    specify the `x` position of the knots (with softmax);
    similarly for the `y` position.
    There will be additional `knots_len` parameters to specify the derivatives
    at knots unless `smooth == True`.

    Note that `knots_len` must be at least equal 2. Also note that

        SplineNet(2, smooth=True)

    is basically an identity net (although it has two dummy parameters!)

    Can be used as a probability distribution convertor for variables with
    nonzero probability in [0, 1].

    Parameters
    ----------
    knots_len : int
        number of knots of the spline.
    xlim & ylim : array-like, optional
        the min and max values for `x` & `y` of the knots.
    knots_x & knots_y & knots_d : None or tensors, optional
        fix corresponding tensors to the input if provided.
    weights_x & weights_y & weights_d : None or tensors, optional
        fix corresponding tensors to the input if provided.
    spline_shape : array-like, optional
        specifies number of splines organized as a tensor
        (default is [], indicating there is only one spline).
    knots_axis : int, optional
        relevant only if spline_shape is not empty list (default value is -1).
    """

    def __init__(self, knots_len, xlim=(0, 1), ylim=(0, 1),
            knots_x=None, knots_y=None, knots_d=None,
            weights_x=None, weights_y=None, weights_d=None,
            spline_shape=[], knots_axis=-1,
            smooth=False, Spline=RQSpline, set_param2zero=True,
            label='spline', **spline_kwargs
            ):
        super().__init__()
        self.label = label

        # knots_len and spline_shape are relevant only if flag is True
        flag = (knots_x is None) or (knots_y is None) or (knots_d is None)

        assert not (flag and knots_len < 2), "oops: knots_len < 2 for splines"

        self.knots_len = knots_len
        self.knots_x = knots_x
        self.knots_y = knots_y
        self.knots_d = knots_d
        self.spline_shape = spline_shape
        self.knots_axis = knots_axis

        self.Spline = Spline
        self.spline_kwargs = dict(**spline_kwargs, knots_axis=knots_axis)

        self.softmax = torch.nn.Softmax(dim=self.knots_axis)
        self.softplus = torch.nn.Softplus(beta=np.log(2))
        # we set the beta of Softplus to log(2) so that self.softplust(0)
        # returns 1. With this setting it would be easy to set the derivatives
        # to 1 (with zero inputs).

        def init(n):
            spline_shape_ = list(spline_shape)
            spline_shape_.insert(knots_axis, n)
            return torch.randn(*spline_shape_) / n**0.5

        if knots_x is None:
            self.xlim, self.xwidth = xlim, xlim[1] - xlim[0]
            if weights_x is None:
                weights_x = torch.nn.Parameter(init(knots_len - 1))
            self.weights_x = weights_x

        if knots_y is None:
            self.ylim, self.ywidth = ylim, ylim[1] - ylim[0]
            if weights_y is None:
                weights_y = torch.nn.Parameter(init(knots_len - 1))
            self.weights_y = weights_y

        if knots_d is None:
            if weights_d is None and (not smooth):
                weights_d = torch.nn.Parameter(init(knots_len))
            self.weights_d = weights_d

        if set_param2zero:
            self.set_param2zero()

    def forward(self, x):
        spline = self.make_spline()
        x_reshaped = x.reshape(*self.spline_shape, -1)
        return spline(x_reshaped).reshape(x.shape)

    def reverse(self, x):
        spline = self.make_spline()
        x_reshaped = x.reshape(*self.spline_shape, -1)
        return spline.reverse(x_reshaped).reshape(x.shape)

    def make_spline(self):
        dim = self.knots_axis
        zero_shape = list(self.spline_shape)
        zero_shape.insert(dim, 1)
        zero = lambda w: torch.zeros(zero_shape, device=w.device)
        cumsumsoftmax = lambda w: torch.cumsum(self.softmax(w), dim=dim)
        to_coord = lambda w: torch.cat((zero(w), cumsumsoftmax(w)), dim=dim)
        to_deriv = lambda d: self.softplus(d) if d is not None else None

        knots_x = self.knots_x
        if knots_x is None:
            knots_x = to_coord(self.weights_x) * self.xwidth + self.xlim[0]

        knots_y = self.knots_y
        if knots_y is None:
            knots_y = to_coord(self.weights_y) * self.ywidth + self.ylim[0]

        knots_d = self.knots_d
        if knots_d is None:
            knots_d = to_deriv(self.weights_d)

        mydict = {'knots_x': knots_x, 'knots_y': knots_y, 'knots_d': knots_d}

        return self.Spline(**mydict, **self.spline_kwargs)

    def set_param2zero(self):
        for param in self.parameters():
            torch.nn.init.zeros_(param)

    def set_param2normal(self, mean=0.0, std=1.0):
        for param in self.parameters():
            torch.nn.init.normal_(param, mean=mean, std=std)
