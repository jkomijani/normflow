# Copyright (c) 2021-2025 Javad Komijani

"""
This module includes several basic subclasses of `torch.nn.Module` that are
designed specifically for scalar tensors. These subclasses implement various
transformations, making them useful for applications in machine learning,
particularly in probabilistic modeling and generative tasks.
"""

# pylint: disable=relative-beyond-top-level, arguments-differ, too-many-locals
# pylint: disable=too-many-arguments, too-many-positional-arguments
# pylint: disable=invalid-name

from typing import Tuple, Union, Sequence, Type, Optional

import torch
import numpy as np

from ...lib.spline import RQSpline
from .convNd import Conv4d


__all__ = [
    "ConvBlock",
    "DenseBlock",
    "ResidualBlock",
    "Affine",
    "Pade32",
    "SplineNet",
    "RQSplineWithGrad",
    "AvgNeighborPool"
]

Number = Union[int, float, complex]
Tensor = torch.Tensor


# =============================================================================
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
            If set to 0, this is interpreted as 1 internally, and the model
            will automatically unsqueeze the input to add a channel axis before
            processing.
        out_channels (int): Number of output channels.
            If set to 0, this is interpreted as 1 internally, and the model
            will automatically squeeze the channel axis from the output after
            processing.
        kernel_size (int or tuple): Size of the convolutional kernel.
        conv_ndim (int, optional): Convolution dimension (default: 2).
        hidden_sizes (Sequence, optional): Sizes of hidden layers
            (default: None).
        norms (Sequence, optional): Normalization layers (default: None).
        acts (Sequence, optional): Activation functions (default: None).
        dropouts (Sequence, optional): Dropout layers (default: None).
        pre_act (optional): Pre-activation layer (default: None).
        **kwargs: All other kwargs to pass to CNN (such as bias).
    """

    conv_map = {
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
        super().__init__()

        # Handle "channel-less" convention by introduced effetive channels
        eff_in_channels = 1 if in_channels == 0 else in_channels
        eff_out_channels = 1 if out_channels == 0 else out_channels

        if hidden_sizes is None:
            sizes = (eff_in_channels, eff_out_channels)
        else:
            sizes = (eff_in_channels, *hidden_sizes, eff_out_channels)

        n_layers = len(sizes) - 1

        norms, acts, dropouts = \
            self._check_accessories(norms, acts, dropouts, n_layers)

        kwargs.update({'padding': 'same', 'padding_mode': 'circular'})

        layers = [] if pre_act is None else [pre_act]

        conv_cls = self.conv_map[conv_ndim]

        for i in range(n_layers):
            layers.append(
                conv_cls(sizes[i], sizes[i + 1], kernel_size, **kwargs)
            )
            for layer in [norms[i], acts[i], dropouts[i]]:
                if layer is not None:
                    layers.append(layer)

        self.layers = torch.nn.Sequential(*layers)
        self.add_input_axis = in_channels == 0
        self.remove_output_axis = out_channels == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape.
            If `add_input_axis` is True, an extra axis is added at dimension 1.

        Returns:
            torch.Tensor: Output tensor after passing through the layers.
            If `remove_output_axis` is True, dimension 1 is squeezed out.
        """
        if self.add_input_axis:
            x = x.unsqueeze(1)

        x = self.layers(x)

        if self.remove_output_axis:
            x = x.squeeze(1)

        return x

    def set_param2zero(self):
        """Set all trainable parameters to zero."""
        for layer in self.layers:
            for param in layer.parameters():
                torch.nn.init.zeros_(param)

    def set_param2normal(self, mean: float = 0.0, std: float = 1.0):
        """Set all trainable parameters to Gaussian with given mean and std."""
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
            If set to 0, this is interpreted as 1 internally, and the model
            will automatically unsqueeze the input to add a feature axis before
            processing.
        - out_features (int): Number of output features.
            If set to 0, this is interpreted as 1 internally, and the model
            will automatically squeeze the feature axis from the output after
            processing.
        - hidden_sizes (Sequence, optional): Sizes of hidden layers
          (default: None).
        - acts (Sequence, optional): Activation functions (default: None).
        - pre_act (optional): Pre-activation layer (default: None).
        - features_axis (int, optional): Th features axis (default: -1).
        - **kwargs: All other kwargs to pass to nn.Linear (such as bias).
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
        # Handle "feature-less" convention by introduced effetive feature axis
        eff_in_features = 1 if in_features == 0 else in_features
        eff_out_features = 1 if out_features == 0 else out_features

        if hidden_sizes is None:
            sizes = (eff_in_features, eff_out_features)
        else:
            sizes = (eff_in_features, *hidden_sizes, eff_out_features)

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
        self.add_input_axis = in_features == 0
        self.remove_output_axis = out_features == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape.
            If `add_input_axis` is True, an extra axis is added at dimension
            specifed by self.feature_axis.

        Returns:
            torch.Tensor: Output tensor after passing through the layers.
            If `remove_output_axis` is True, dimension seelf.feature_axis is
            squeezed out.
        """
        features_axis = self.features_axis
        if self.add_input_axis:
            x = x.unsqueeze(features_axis)

        if features_axis == -1:
            x = self.layers(x)
        else:
            x = torch.movedim(x, features_axis, -1)
            x = self.layers(x)
            x = torch.movedim(x, -1, features_axis)

        if self.remove_output_axis:
            x = x.squeeze(features_axis)

        return x

    def set_param2zero(self):
        """Set all trainable parameters to zero."""
        for layer in self.layers:
            for param in layer.parameters():
                torch.nn.init.zeros_(param)

    def set_param2normal(self, mean: float = 0.0, std: float = 1.0):
        """Set all trainable parameters to Gaussian with given mean and std."""
        for layer in self.layers:
            for param in layer.parameters():
                torch.nn.init.normal_(param, mean=mean, std=std)


class ResidualBlock(torch.nn.Module):
    """
    Pre-activation Residual Block with flexible conv, norm, and activation.

    Implements the pre-activation style of ResNet (He et al., 2016), where
    normalization and activation are applied before each convolution. The
    skip connection is either an identity or a 1×1 projection if input and
    output channels differ.

    Features:
        - Supports 1D, 2D, 3D, or 4D convolutions.
        - Flexible normalization (BatchNorm, GroupNorm, etc.).
        - Flexible activation (ReLU, SiLU, LeakyReLU, etc.).
        - Optional skip projection if channels differ.

    Args:
        in_channels (int): Number of input channels.
            If set to 0, this is interpreted as 1 internally, and the model
            will automatically unsqueeze the input to add a channel axis before
            processing.
        out_channels (int): Number of output channels.
            If set to 0, this is interpreted as 1 internally, and the model
            will automatically squeeze the channel axis from the output after
            processing.
        kernel_size (int): Kernel size of convolutions.
        conv_ndim (int, default=2): Convolution dimension (1,2,3,[4]).
        norm_cls (nn.Module, optional): Normalization class. Defaults to
            BatchNorm of the appropriate dimension.
        act_cls (nn.Module, optional): Activation class. Defaults to SiLU.
        **kwargs: Additional args passed to convolution layers.
    """

    conv_map = {
        1: torch.nn.Conv1d,
        2: torch.nn.Conv2d,
        3: torch.nn.Conv3d,
        4: Conv4d  # replace with actual 4D conv if available
    }

    default_norm_map = {
        1: torch.nn.BatchNorm1d,
        2: torch.nn.BatchNorm2d,
        3: torch.nn.BatchNorm3d,
        4: None  # replace with 4D norm if available
    }

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        conv_ndim: int = 2,
        mid_channels: int | None = None,
        norm_cls: Optional[Type[torch.nn.Module]] = None,
        act_cls: Optional[Type[torch.nn.Module]] = torch.nn.SiLU,
        **kwargs
    ):
        super().__init__()

        assert conv_ndim in (1, 2, 3, 4), "conv_ndim must be 1,2,3,4"
        conv_cls = self.conv_map[conv_ndim]

        norm_cls = norm_cls or self.default_norm_map[conv_ndim]

        # Handle "channel-less" convention by introduced effetive channels
        eff_in_channels = 1 if in_channels == 0 else in_channels
        eff_out_channels = 1 if out_channels == 0 else out_channels

        # Pre-activation conv blocks
        mid_channels = mid_channels or out_channels
        kwargs.update({'padding': 'same', 'padding_mode': 'circular'})
        self.conv_block1 = torch.nn.Sequential(
            norm_cls(eff_in_channels),
            act_cls(inplace=True),
            conv_cls(eff_in_channels, mid_channels, kernel_size, **kwargs)
        )
        self.conv_block2 = torch.nn.Sequential(
            norm_cls(mid_channels),
            act_cls(inplace=True),
            conv_cls(mid_channels, eff_out_channels, kernel_size, **kwargs)
        )

        # Skip connection
        if eff_in_channels != eff_out_channels:
            self.skip = conv_cls(
                eff_in_channels, eff_out_channels, kernel_size=1
            )
        else:
            self.skip = torch.nn.Identity()

        self.add_input_axis = in_channels == 0
        self.remove_output_axis = out_channels == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape.
            If `add_input_axis` is True, an extra axis is added at dimension 1.

        Returns:
            torch.Tensor: Output tensor after passing through the layers.
            If `remove_output_axis` is True, dimension 1 is squeezed out.
        """
        if self.add_input_axis:
            x = x.unsqueeze(1)

        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = out + self.skip(x)

        if self.remove_output_axis:
            out = out.squeeze(1)

        return out


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        scale, bias = self.get_parameters_reshaped(x.shape)
        return scale * x + bias

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """Reverse pass."""
        scale, bias = self.get_parameters_reshaped(y.shape)
        return (y - bias) / scale

    def get_parameters_reshaped(self, shape):
        """Compute paramaters and reshape if required."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        a = self.get_parameters_reshaped(x.shape)  # a is derivative at x = 0
        y = a * x * (a + x**2) / (1 + a * x**2)
        return y

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """Reverse pass."""
        a = self.get_parameters_reshaped(y.shape)  # a is derivative at x = 0
        x = self.reverse_pade32(y / a, a)
        return x

    def get_parameters_reshaped(self, shape):
        """Compute paramaters and reshape if required."""
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


class SplineNet(torch.nn.Module):
    """
    Neural network wrapper for learnable spline-based transformations.

    The network defines a monotonic mapping from `xlim` to `ylim` using a
    trainable spline function. The specific spline implementation can be
    provided via the `Spline` argument. By default, it uses a rational
    quadratic spline (`RQSpline`).

    The number of knots is specified by `knots_len`. The first knot is fixed
    at (xlim[0], ylim[0]) and the last at (xlim[1], ylim[1]). The coordinates
    of intermediate knots are learned unless explicitly provided through
    `knots_x` or `knots_y`.

    If `knots_x` is None, `(knots_len - 1)` parameters are learned to define
    the x-positions of the knots via a softmax; the same applies to y.
    Additional `knots_len` parameters control the derivatives at the knots,
    unless `smooth=True`.

    When `xlim=(0, 1)` and `ylim=(0, 1)`, the transformation becomes a smooth
    bijection from [0, 1] to [0, 1], making it suitable for applications such
    as normalizing flows or differentiable coordinate transforms.

    Notes
    -----
    - `knots_len` must be at least 2.
    - `SplineNet(2, smooth=True)` yields an identity-like mapping with two
      dummy parameters.

    Parameters
    ----------
    knots_len : int
        Number of knots in the spline.
    xlim, ylim : array-like, optional
        The minimum and maximum values for x and y coordinates of the knots.
        Defaults to [0, 1].
    knots_x, knots_y, knots_d : torch.Tensor or None, optional
        Fixed knot positions or derivatives, if provided.
    weights_x, weights_y, weights_d : torch.Tensor or None, optional
        Fixed weight tensors that override the default initialization.
    spline_shape : array-like or None, optional
        Shape of the tensor specifying the number of independent splines.
        Defaults to None, indicating a single spline.
    knots_axis : int, optional
        Relevant only if `spline_shape` is not an empty list. Default is -1.
    smooth : bool, optional
        If True, enforces smooth derivatives across knots. Default is False.
    Spline : callable, optional
        Spline class or factory to use. Defaults to `RQSpline`.
    set_param2zero : bool, optional
        If True, initializes internal parameters to zero. Defaults to True.
    **spline_kwargs : dict
        Additional keyword arguments passed to the `Spline` constructor.
    """
    def __init__(
        self,
        knots_len,
        xlim=(0, 1), ylim=(0, 1),
        knots_x=None, knots_y=None, knots_d=None,
        weights_x=None, weights_y=None, weights_d=None,
        spline_shape=None, knots_axis=-1, smooth=False, Spline=RQSpline,
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
        self.spline_shape = spline_shape or []
        self.knots_axis = knots_axis

        self.Spline = Spline
        self.spline_kwargs = {"knots_axis": knots_axis, **spline_kwargs}

        # Softmax for normalized x/y weights, Softplus for positive derivatives
        self.softmax = torch.nn.Softmax(dim=self.knots_axis)
        self.softplus = torch.nn.Softplus(beta=np.log(2))
        # Softplus(0) = 1 when beta = log(2); this makes it easy to initialize
        # derivatives at 1 (identity slope) with zero-weight initialization.

        def init(n):
            """Utility to initialize a tensor of size (spline_shape, n)."""
            spline_shape_ = list(self.spline_shape)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward spline transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of values within the spline domain (`xlim`), assuming
            the spline is not defined outside this range.

        Returns
        -------
        torch.Tensor
            Transformed tensor of the same shape, mapping values from `xlim`
            to the corresponding range defined by `ylim`.
        """
        # Build the spline transformation defined by current parameters.
        spline = self.make_spline()

        # Reshape input to match the spline shape for evaluation.
        x_reshaped = x.reshape(*self.spline_shape, -1)

        # Evaluate spline, reshape outputs back to the original shape, & return
        return spline(x_reshaped).reshape(x.shape)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the inverse spline transformation.

        Parameters
        ----------
        y : torch.Tensor
            Input tensor of values within the spline range (`ylim`), assuming
            the spline is not defined outside this range.

        Returns
        -------
        torch.Tensor
            Inverse-transformed tensor of the same shape, mapping values from
            `ylim` back to the corresponding domain defined by `xlim`.
        """
        # Build the spline transformation defined by current parameters.
        spline = self.make_spline()

        # Reshape input to match the spline shape for evaluation.
        y_reshaped = y.reshape(*self.spline_shape, -1)

        # Evaluate reverse spline, reshape outputs, & return
        return spline.reverse(y_reshaped).reshape(y.shape)

    def make_spline(self):
        """Make an spline for forward and reverse passes."""
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
        """Set all trainable parameters to zero."""
        for param in self.parameters():
            torch.nn.init.zeros_(param)

    def set_param2normal(self, mean: float = 0.0, std: float = 1.0):
        """Set all trainable parameters to Gaussian with given mean and std."""
        for param in self.parameters():
            torch.nn.init.normal_(param, mean=mean, std=std)


class RQSplineWithGrad(SplineNet, torch.nn.Module):
    """
    Extension of `SplineNet` that also computes the first-order derivative
    (gradient) of the rational quadratic (RQ) spline transformation.

    The superclass `SplineNet` already defines a monotonic RQ spline mapping
    from (0, 0) to (1, 1). This subclass behaves identically, except that its
    `forward` and `reverse` methods also return the derivative of the
    transformation with respect to the input.

    The mapping is strictly monotonic increasing and can be used as a smooth
    bijection over [0, 1].

    For changing the spline configuration (number of knots, domain/range,
    smoothness), see the documentation of `SplineNet`.
    """

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the forward spline transformation and its gradient.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of values within [0, 1].

        Returns
        -------
        y : torch.Tensor
            Transformed values of the spline at `x` in range [0, 1].
        grad : torch.Tensor
            Gradient (derivative) of the transformation at `x`.
        """
        # Build the spline transformation defined by current parameters.
        spline = self.make_spline()

        # Reshape input to match the spline shape for evaluation.
        x_reshaped = x.reshape(-1)

        # Evaluate spline and compute its gradient.
        y, grad = spline(x_reshaped, grad=True)

        # Reshape outputs back to the original input shape.
        y, grad = y.reshape(x.shape), grad.reshape(x.shape)

        return y, grad

    def reverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the inverse spline transformation and its gradient.

        Parameters
        ----------
        y : torch.Tensor
            Input tensor of values within [0, 1].

        Returns
        -------
        x : torch.Tensor
            Inverse-transformed values of the spline at `y` in range [0, 1].
        grad : torch.Tensor
            Gradient (derivative) of the inverse transformation at `y`.
        """
        # Build the spline transformation defined by current parameters.
        spline = self.make_spline()

        # Reshape input to match the spline shape for evaluation.
        y_reshaped = y.reshape(-1)

        # Evaluate inverse spline and compute its gradient.
        x, grad = spline.reverse(y_reshaped, grad=True)

        # Reshape outputs back to the original input shape.
        x, grad = x.reshape(x.shape), grad.reshape(x.shape)

        return x, grad


class AvgNeighborPool(torch.nn.Module):
    """Computes the mean of neighboring elements along non-batch dimensions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return neighbor_mean(x, dim=range(1, x.ndim))


class Abs(torch.nn.Module):
    """Introduced for adding to the list of activations"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
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
    """
    Retrieve an activation function.

    Args:
        act (str | torch.nn.Module | None):
            - If None, returns an identity mapping (`torch.nn.Identity`).
            - If str, looks up the activation class in `activations_dict` and
              instantiates it.
            - If already a `torch.nn.Module`, returns it directly.

    Returns:
        torch.nn.Module: The corresponding activation function.
    """
    if act is None:
        return torch.nn.Identity()

    if isinstance(act, str):
        return activations_dict[act]()

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
