# Copyright (c) 2021-2025 Javad Komijani

"""
This module includes several basic subclasses of `torch.nn.Module` that are
designed specifically for scalar tensors. These subclasses implement various
transformations, making them useful for applications in machine learning,
particularly in probabilistic modeling and generative tasks.
"""


import torch
import numpy as np
from typing import Union, Sequence

from ...lib.spline import RQSpline
from .convNd import Conv4d


Number = Union[int, float, complex]
Tensor = torch.Tensor


class ConvBlock(torch.nn.Module):
    r"""
    A flexible convolutional module extending PyTorch's convolutional layers.

    This module supports up to 4D convolutions with optional hidden layers,
    normalizations, activations, and dropout modules for each layer.

    Instantiating with default options is equivalent to `torch.nn.Conv2d`
    with `padding='same'` and `padding_mode='circular'`.

    The input and output are tensors of 3+ dimensions with the signature:
    `tensor(:, ch, ...)`, where `:` represents the batch, `ch` the channels,
    and `...` the feature axes.

    .. math::
        out(:, ch_o, ...) = bias(ch_o) +
                        \sum_{ch_i} weight(ch_o, ...) \star input(:, ch_i, ...)

    where :math:`\star` is the n-dimensional cross-correlation operator.
    Supported feature dimensions: 1, 2, 3, and 4. The channels axis can be
    customized.

    The optional `hidden_sizes` can be a sequence specifying hidden layer
    sizes. Similarly, `norms`, `acts`, and `dropouts` must match `hidden_sizes`
    plus 1 in length if provided, containing appropriate modules or None.
    Finally, use `pre_act` for pre-activations.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolutional kernel.
        conv_ndim (int, optional): Convolution dimension (default: 2).
        hidden_sizes (Sequence, optional): Sizes of hidden layers
           (default: None).
        norms (Sequence, optional): Normalization layers (default: None).
        acts (Sequence, optional): Activation functions (default: None).
        dropouts (Sequence, optional): Dropout layers (default: None).
        pre_act (optional): Pre-activation layer (default: None).
    """

    _conv = {
        1: torch.nn.Conv1d,
        2: torch.nn.Conv2d,
        3: torch.nn.Conv3d,
        4: Conv4d
    }

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        conv_ndim: int = 2,
        hidden_sizes: Sequence[int] = None,
        norms=None,
        acts=None,
        dropouts=None,
        pre_act=None,
        **kwargs  # all other kwargs to pass to torch.nn.Conv?d
    ):

        if hidden_sizes is None:
            sizes = (in_channels, out_channels)
        else:
            sizes = (in_channels, *hidden_sizes, out_channels)

        n_layers = len(sizes) - 1

        norms, acts, dropouts = \
            self._check_accessories(norms, acts, dropouts, n_layers)

        kwargs = dict(padding='same', padding_mode='circular')
        kwargs.update(kwargs)

        layers = [] if pre_act is None else [pre_act]

        conv = self._conv[conv_ndim]

        for i in range(n_layers):
            layers.append(conv(sizes[i], sizes[i+1], kernel_size, **kwargs))
            for layer in [norms[i], acts[i], dropouts[i]]:
                if layer is not None:
                    layers.append(layer)

        super().__init__()
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def set_param2zero(self):
        for layer in self.layers:
            for param in layer.parameters():
                torch.nn.init.zeros_(param)

    def set_param2normal(self, mean=0.0, std=1.0):
        for layer in self.layers:
            for param in layer.parameters():
                torch.nn.init.normal_(param, mean=mean, std=std)

    @staticmethod
    def _check_accessories(norms, acts, dropouts, n_layers):

        if norms is None:
            norms = [None] * n_layers
        else:
            assert len(norms) == n_layers

        if acts is None:
            acts = [None] * n_layers
        else:
            assert len(acts) == n_layers

        if dropouts is None:
            dropouts = [None] * n_layers
        else:
            assert len(dropouts) == n_layers

        return norms, acts, dropouts


class DenseBlock(torch.nn.Module):
    """
    A flexible extension to `torch.nn.Linear` with possible hidden layers and
    activations.

    The optional `hidden_sizes` can be a sequence specifying hidden layer
    sizes. Similarly, `acts` must match `hidden_sizes` plus 1 in length if
    provided, containing appropriate modules or None. Finally, use `pre_act`
    for pre-activations.

    The axes of the input and output tensors are treated by default as
    `(..., f)`, where `...` stands for any number of dimensions and `f` for the
    features axis. However, it is possible to change the features axis from
    last to any other axis.

    Args:
        - in_features (int): Number of input features.
        - out_features (int): Number of output features.
        - hidden_sizes (Sequence, optional): Sizes of hidden layers
          (default: None).
        - acts (Sequence, optional): Activation functions (default: None).
        - pre_act (optional): Pre-activation layer (default: None).
        - features_axis (int, optional): Th features axis (default: -1).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_sizes: Sequence[int] = None,
        acts=None,
        pre_act=None,
        features_axis: int = -1,
        **kwargs  # all other kwargs to pass to torch.nn.Linear
    ):

        if hidden_sizes is None:
            sizes = (in_features, out_features)
        else:
            sizes = (in_features, *hidden_sizes, out_features)

        n_layers = len(sizes) - 1

        if acts is None:
            acts = [None] * n_layers
        else:
            assert len(acts) == n_layers

        layers = [] if pre_act is None else [pre_act]

        Linear = torch.nn.Linear

        for i in range(n_layers):
            layers.append(Linear(sizes[i], sizes[i+1], **kwargs))
            if acts[i] is not None:
                layers.append(acts[i])

        super().__init__()

        self.layers = torch.nn.Sequential(*layers)
        self.features_axis = features_axis

    def forward(self, x):
        features_axis = self.features_axis
        if features_axis == -1:
            return self.layers(x)
        else:
            x = torch.movedim(x, features_axis, -1)
            x = self.layers(x)
            return torch.movedim(x, -1, features_axis)

    def set_param2zero(self):
        for layer in self.layers:
            for param in layer.parameters():
                torch.nn.init.zeros_(param)

    def set_param2normal(self, mean=0.0, std=1.0):
        for layer in self.layers:
            for param in layer.parameters():
                torch.nn.init.normal_(param, mean=mean, std=std)


class Affine(torch.nn.Module):
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

    def __init__(self,
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

    def forward(self, x):
        scale, bias = self.get_parameters_reshaped(x.shape)
        return scale * x + bias

    def reverse(self, y):
        scale, bias = self.get_parameters_reshaped(y.shape)
        return (y - bias) / scale

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


class Pade32(torch.nn.Module):
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

    def forward(self, x):
        a = self.get_parameters_reshaped(x.shape)  # a is derivative at x = 0
        y = a * x * (a + x**2) / (1 + a * x**2)
        return y

    def reverse(self, y):
        a = self.get_parameters_reshaped(y.shape)  # a is derivative at x = 0
        x = self.reverse_pade32(y / a, a)
        return x

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

    def __init__(
        self,
        knots_len,
        xlim=(0, 1), ylim=(0, 1),
        knots_x=None, knots_y=None, knots_d=None,
        weights_x=None, weights_y=None, weights_d=None,
        spline_shape=[], knots_axis=-1, smooth=False, Spline=RQSpline,
        set_param2zero=True,
        **spline_kwargs
    ):
        super().__init__()

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


class AvgNeighborPool(torch.nn.Module):
    """Computes the mean of neighboring elements along non-batch dimensions."""

    def forward(self, x):
        return neighbor_mean(x, dim=range(1, x.ndim))


class Abs(torch.nn.Module):
    """Introduced for adding to the list of activations"""

    def forward(self, x):
        return torch.abs(x)


activations_dict = {
    'tanh': torch.nn.Tanh,
    'relu': torch.nn.ReLU,
    'silu': torch.nn.SiLU,
    'leaky_relu': torch.nn.LeakyReLU,
    'softplus': torch.nn.Softplus,
    'avg_neighbor_pool': AvgNeighborPool,
    'abs': Abs,
    'none': torch.nn.Identity
}


def get_activation(act):

    if act is None:
        return torch.nn.Identity()

    elif isinstance(act, str):
        return activations_dict[act]()

    else:
        return act


def neighbor_mean(x: Tensor, dim: Sequence[int] = None) -> Tensor:
    """
    Computes the mean of neighboring elements along specified dimensions.

    Args:
        - x (Tensor): The input tensor.
        - dim (Sequence[int], optional): The dimensions along which to compute
          the mean. If `None`, all non-batch dimensions are used.

    Returns:
        Tensor: A tensor of the same shape as `x`, containing the mean of
        neighboring elements. If all specified dimensions are singleton
        (size 1), `x` is returned unchanged.
    """

    if dim is None:
        dim = range(1, x.ndim)  # Exclude batch axis (dim 0)

    y, ndim = torch.zeros_like(x), 0

    for mu in dim:
        if x.shape[mu] > 1:  # Only compute for non-singleton dimensions
            y += torch.roll(x, 1, mu) + torch.roll(x, -1, mu)
            ndim += 1

    return x if ndim == 0 else y / (2 * ndim)
