# Copyright (c) 2021-2025 Javad Komijani

"""
This module defines mask-based invertible coupling transformations that
are subclasses of `Module_` and compute the logarithm of the Jacobian
determinant as part of their forward and reverse passes.

The base class `Coupling_` implements the partitioning and sequencing
logic for coupling layers, while leaving the per-layer transformation
(`atomic_forward` and `atomic_reverse`) to be implemented by subclasses.

Available subclasses include:
- `AdditiveCoupling_` : Applies shift-only transformations.
- `AffineCoupling_`   : Applies scale-and-shift (affine) transformations.
- `RQSplineCoupling_` : Applies rational-quadratic spline transformations.

As in `Module_`, the trailing underscore in the class names indicates
that the transformation methods (`forward` and `reverse`) both return
the transformed tensor and the accumulated log-Jacobian determinant.
"""

# pylint: disable=relative-beyond-top-level, arguments-differ, too-many-locals
# pylint: disable=too-many-arguments, too-many-positional-arguments
# pylint: disable=invalid-name

from abc import abstractmethod, ABC

import torch

from .._core import Module_
from ...lib.spline import make_rq_spline_field


# =============================================================================
class Coupling_(Module_, ABC):
    """
    Base class for a sequence of invertible, mask-based coupling layers.

    The input is split into two complementary partitions using a mask. In each
    coupling layer, one partition (active) is transformed while the other
    (frozen) remains unchanged. The transformation applied to the active
    partition is conditioned on the frozen partition via a feature map network.
    Across successive layers, the roles of the partitions are swapped, and
    this alternation enables efficient computation of the Jacobian determinant.

    Subclasses define the actual per-layer transformation by implementing
    `atomic_forward` (forward direction) and `atomic_reverse` (inverse
    direction). This base class manages the masking, partition swapping,
    and sequencing of layers.

    Parameters
    ----------
    feature_map_nets : list[torch.nn.Module or None]
        List of neural networks that parameterize the coupling layers.
        - Each entry corresponds to one layer.
        - If an entry is `None`, that layer acts as identity (no
          transformation is applied).
        - The network outputs must match the parameterization required
          by the subclass’s transformation (e.g., `(t, s)` for affine).

    mask : object
        Mask defining how the input is split into two partitions.

    channels_axis : int, default=1
        Axis corresponding to the channel dimension. A singleton channels axis
        is automatically added to inputs before processing and removed
        afterwards when `handle_channel_axis=True`.

    handle_channel_axis : bool, default=True
        If True, a singleton channel axis is automatically added to inputs
        before processing and removed afterwards. This allows coupling layers
        to operate in both channel-free and channel-based architectures.

    Notes
    -----
    - `atomic_forward` implements the forward transformation of the active
      partition conditioned on the frozen one.
    - `atomic_reverse` implements the corresponding inverse transformation.
    - Alternating the active/frozen partitions across layers ensures that all
      input dimensions are eventually transformed.
    - Efficient log-Jacobian accumulation is supported by design.
    """

    def __init__(
        self,
        feature_map_nets,
        mask,
        channels_axis: int = 1,
        handle_channel_axis: bool = True
    ):
        super().__init__()
        self.feature_map_nets = torch.nn.ModuleList(feature_map_nets)
        self.mask = mask
        self.channels_axis = channels_axis
        self.handle_channel_axis = handle_channel_axis

    @property
    def nets(self):
        """List of neural nets used to parameterize coupling layers."""
        return self.feature_map_nets

    def forward(self, x, log0=0):
        """
        Apply the forward transformation through all coupling layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be transformed.
        log0 : float or torch.Tensor, optional
            Initial log-Jacobian determinant to accumulate into. Default is 0.

        Returns
        -------
        torch.Tensor
            Transformed tensor after all coupling layers.
        float or torch.Tensor
            Updated log-Jacobian determinant.
        """
        # Split input into two partitions & add channel axis if needed
        x = list(self.mask.split(x))  # x = [x_0, x_1]

        if self.handle_channel_axis:
            x = [a.unsqueeze(self.channels_axis) for a in x]

        # Apply each coupling layer in sequence
        for k, net in enumerate(self.feature_map_nets):
            if net is None:
                continue
            parity = k % 2  # Decide which partition is active
            x[parity], log0 = self.atomic_forward(
                x_active=x[parity],
                x_frozen=x[1 - parity],
                parity=parity,
                net=net,
                log0=log0
            )

        if self.handle_channel_axis:
            x = [a.squeeze(self.channels_axis) for a in x]

        return self.mask.cat(*x), log0

    def reverse(self, x, log0=0):
        """
        Apply the reverse transformation through all coupling layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be inverted.
        log0 : float or torch.Tensor, optional
            Initial log-Jacobian determinant to accumulate into. Default
            is 0.

        Returns
        -------
        torch.Tensor
            Inverse-transformed tensor.
        float or torch.Tensor
            Updated log-Jacobian determinant.
        """
        # Split input into two partitions & add channel axis if needed
        x = list(self.mask.split(x))  # x = [x_0, x_1]
        if self.handle_channel_axis:
            x = [a.unsqueeze(self.channels_axis) for a in x]

        n_layers = len(self.feature_map_nets)

        # Apply coupling layers in reverse order
        for k, net in enumerate(reversed(self.feature_map_nets)):
            if net is None:
                continue
            parity = (n_layers - 1 - k) % 2
            x[parity], log0 = self.atomic_reverse(
                x_active=x[parity],
                x_frozen=x[1 - parity],
                parity=parity,
                net=net,
                log0=log0
            )

        if self.handle_channel_axis:
            x = [a.squeeze(self.channels_axis) for a in x]

        return self.mask.cat(*x), log0

    @abstractmethod
    def atomic_forward(self, *, x_active, x_frozen, parity, net, log0=0):
        """
        Apply the forward transformation to the active partition for a
        single coupling layer.

        Must be implemented by subclasses.

        Parameters
        ----------
        x_active : torch.Tensor
            Active partition of the input.
        x_frozen : torch.Tensor
            Frozen partition of the input.
        parity : int
            Index indicating which partition is active (0 or 1).
        net : torch.nn.Module
            Neural network parameterizing the transformation.
        log0 : float or torch.Tensor
            Accumulated log-Jacobian.

        Returns
        -------
        torch.Tensor
            Transformed active partition.
        float or torch.Tensor
            Updated log-Jacobian.
        """

    @abstractmethod
    def atomic_reverse(self, *, x_active, x_frozen, parity, net, log0=0):
        """
        Apply the reverse transformation to the active partition for a
        single coupling layer.

        Must be implemented by subclasses.

        Parameters
        ----------
        x_active : torch.Tensor
            Active partition of the input.
        x_frozen : torch.Tensor
            Frozen partition of the input.
        parity : int
            Index indicating which partition is active (0 or 1).
        net : torch.nn.Module
            Neural network parameterizing the transformation.
        log0 : float or torch.Tensor
            Accumulated log-Jacobian.

        Returns
        -------
        torch.Tensor
            Inverse-transformed active partition.
        float or torch.Tensor
            Updated log-Jacobian.
        """


# =============================================================================
class AdditiveCoupling_(Coupling_):
    """
    A `Coupling_` subclass implementing shift-only (additive) transformations.

    In each coupling layer, the frozen partition `x_frozen` is passed through
    a subnetwork `net`, whose output is interpreted as a shift `t`. The active
    partition `x_active` is then updated as:

        y = x_active + t

    This operation has a unit Jacobian determinant, so the accumulated
    log-Jacobian remains unchanged.

    Notes
    -----
    - The subnetwork `net` must output a tensor matching the shape of the
      active partition.
    - Applying `mask.purify` ensures that only the active partition is
      transformed, while the frozen partition remains unchanged.
    """

    def atomic_forward(self, *, x_active, x_frozen, parity, net, log0=0):
        t = net(x_frozen)
        return self.mask.purify(x_active + t, channel=parity), log0

    def atomic_reverse(self, *, x_active, x_frozen, parity, net, log0=0):
        t = net(x_frozen)
        return self.mask.purify(x_active - t, channel=parity), log0


# =============================================================================
class AffineCoupling_(Coupling_):
    """
    A `Coupling_` subclass implementing affine scale-and-shift transformations.

    In each coupling layer, the frozen partition `x_frozen` is passed through
    a subnetwork `net`. Its output is split along the channel axis into two
    parts `(t, s)`, which parameterize the affine transformation of the active
    partition `x_active`:

        y = x_active * exp(-|s|) + t

    The absolute value ensures that the scaling factor `exp(-|s|)` never
    exceeds 1, which stabilizes training, especially when stacking many
    affine coupling layers. The log-determinant of the Jacobian is updated
    accordingly at each step.

    Notes
    -----
    - The subnetwork `net` must output a tensor with twice as many channels
      as the active partition. The first half corresponds to the shift `t`,
      and the second half to the scale parameter `s`.
    - Applying `mask.purify` ensures that only the active partition is
      transformed, while the frozen partition remains unchanged.
    - The sign convention differs slightly from standard affine coupling:
      the scaling factor is defined as `exp(-|s|)` instead of `exp(s)`,
      ensuring bounded magnitudes for stability.
    """

    def atomic_forward(self, *, x_active, x_frozen, parity, net, log0=0):
        t, s = net(x_frozen).chunk(2, dim=self.channels_axis)
        t = self.mask.purify(t, channel=parity)
        s = self.mask.purify(s, channel=parity)
        s = s.abs()
        return t + x_active * torch.exp(-s), log0 - self.sum_density(s)

    def atomic_reverse(self, *, x_active, x_frozen, parity, net, log0=0):
        t, s = net(x_frozen).chunk(2, dim=self.channels_axis)
        t = self.mask.purify(t, channel=parity)
        s = self.mask.purify(s, channel=parity)
        s = s.abs()
        return (x_active - t) * torch.exp(s), log0 + self.sum_density(s)


# =============================================================================
class RQSplineCoupling_(Coupling_):
    """
    A `Coupling_` subclass implementing rational quadratic spline
    transformations.

    In each coupling layer, the active partition is transformed by an
    invertible rational quadratic spline, parameterized by a feature map
    network applied to the frozen partition. The Jacobian determinant is
    tracked via the spline derivative.

    Configuration
    -------------
    The following options control spline behavior:
    - `xlim`, `ylim`: Input/output ranges of the spline.
    - `knots_x`, `knots_y`: Knot positions along the input/output axes.
    - `extrap`: Extrapolation behavior outside the specified ranges.

    See `RQSpline` for full details of these options.

    Example
    -------
    To apply linear extrapolation on the right boundary and anti-periodic
    extrapolation on the left boundary:

    >>> extrap = {"left": "anti", "right": "linear"}

    Notes
    -----
    - The network `net` must output a feature map sufficient to parameterize
      the spline. The number of knots is inferred from this output.
    - Applying `mask.purify` ensures that only the active partition is
      transformed, while the frozen partition remains unchanged.
    """

    def __init__(
        self,
        nets,
        mask,
        xlim=(0, 1),
        ylim=(0, 1),
        knots_x=None,
        knots_y=None,
        extrap=None,
        channels_axis=1
    ):

        super().__init__(nets, mask=mask, channels_axis=channels_axis)

        self.spline_kwargs = {
            'xlim': xlim,
            'ylim': ylim,
            'knots_axis': channels_axis,
            'knots_x': knots_x,
            'knots_y': knots_y,
            'extrap': extrap
        }

    def make_rq_spline_field(self, feature_map):
        """Calls `make_rq_spline_field` with the saved key-word argumentes."""
        return make_rq_spline_field(feature_map, **self.spline_kwargs)

    def atomic_forward(self, *, x_active, x_frozen, parity, net, log0=0):
        feature_map = net(x_frozen)
        spline = self.make_rq_spline_field(feature_map)

        # below g is the gradient of spline @ x_active
        x_active, g = spline(x_active, grad=True)

        x_active = self.mask.purify(x_active, channel=parity)
        logg = self.mask.purify(torch.log(g), channel=parity)

        return x_active, log0 + self.sum_density(logg)

    def atomic_reverse(self, *, x_active, x_frozen, parity, net, log0=0):
        feature_map = net(x_frozen)
        spline = self.make_rq_spline_field(feature_map)

        # below g is the gradient of spline @ x_active
        x_active, g = spline(x_active, grad=True)

        x_active = self.mask.purify(x_active, channel=parity)
        logg = self.mask.purify(torch.log(g), channel=parity)

        return x_active, log0 + self.sum_density(logg)

    def _hack(self, *, x_active, x_frozen, parity, net):
        feature_map = net(x_frozen)
        spline = self.make_rq_spline_field(feature_map)
        x_active, g = spline(x_active, grad=True)
        x_active = self.mask.purify(x_active, channel=parity)
        logg = self.mask.purify(torch.log(g), channel=parity)
        return spline, x_active, logg


# =============================================================================
class MultiRQSplineCoupling_(Coupling_):
    """
    A `Coupling_` subclass implementing multiple rational quadratic spline
    transformations, each actiong on an additional channel of the input data.

    In addition to the arguments and option of Coupling_, there are specific
    options for MultiRQSplineCoupling_, which are very similar to those of
    RQSplineCoupling_, except that, e.g., instead of `xlim` here we have
    `xlims`, which is a list. By default the list have two elements, indicating
    there are two rational quadratic splines.

    For more details on using these options see RQSplineCoupling_ and RQSpline.
    """

    def __init__(
        self,
        nets,
        mask,
        xlims=((0, 1), (0, 1)),
        ylims=((0, 1), (0, 1)),
        knots_x=(None, None),
        knots_y=(None, None),
        extraps=(None, None),
        channels_axis=1
    ):

        super().__init__(nets, mask, channels_axis, handle_channel_axis=False)

        self.num_splines = len(xlims)

        self.spline_kwargs_list = [
            {'xlim': xlims[i],
             'ylim': ylims[i],
             'knots_axis': channels_axis,
             'knots_x': knots_x[i],
             'knots_y': knots_y[i],
             'extrap': extraps[i]
             } for i in range(self.num_splines)
        ]

    def atomic_forward(self, *, x_active, x_frozen, parity, net, log0=0):
        feature_map = net(x_frozen)
        spline = self.make_multi_rq_spline_field(feature_map)
        # below g is the gradient of spline @ x_active
        fx_active, g = self.apply_spline(self.preprocess(x_active), spline)
        fx_active, g = self.postprocess(fx_active), self.postprocess(g)
        fx_active = self.mask.purify(fx_active, channel=parity)
        logg = self.mask.purify(torch.log(g), channel=parity)
        return fx_active, log0 + self.sum_density(logg)

    def atomic_reverse(self, *, x_active, x_frozen, parity, net, log0=0):
        feature_map = net(x_frozen)
        spline = self.make_multi_rq_spline_field(feature_map)
        # below g is the gradient of spline @ x_active
        fx_active, g = self.apply_spline(
            self.preprocess(x_active), spline, reverse=True
        )
        fx_active, g = self.postprocess(fx_active), self.postprocess(g)
        fx_active = self.mask.purify(fx_active, channel=parity)
        logg = self.mask.purify(torch.log(g), channel=parity)
        return fx_active, log0 + self.sum_density(logg)

    def preprocess(self, x):
        """Split the x_active to a list of tensors."""
        xs = torch.tensor_split(
            x, sections=self.num_splines, dim=self.channels_axis
        )
        return xs

    def postprocess(self, xs):
        """Concatenate a list of x_active channels into a single tensor."""
        x = torch.cat(xs, dim=self.channels_axis)
        return x

    def make_multi_rq_spline_field(self, feature_map):
        """Calls `make_rq_spline_field` with the saved key-word argumentes."""
        features = torch.tensor_split(
            feature_map, sections=self.num_splines, dim=self.channels_axis
        )
        splines = []
        for feature, kwargs in zip(features, self.spline_kwargs_list):
            splines.append(make_rq_spline_field(feature, **kwargs))
        return splines

    def apply_spline(self, x_actives, splines, reverse=False):
        """Apply each spline to each channel of x_actives."""
        x_actives_out = []
        gs = []
        for i, x_active in enumerate(x_actives):
            transformation = splines[i].reverse if reverse else splines[i]
            x_active, g = transformation(x_active, grad=True)
            x_actives_out.append(x_active)
            gs.append(g)
        return x_actives_out, gs
