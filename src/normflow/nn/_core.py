# Copyright (c) 2021-2026 Javad Komijani

"""
This module contains subclass of `torch.nn.Module` designed for creating
invertible transformations that also compute the logarithm of the Jacobian
of the transformation.

In orther to distinguish the type of the modules, we use a trailing underscore
in the class name to indicate that the `forward` method not only returns the
transformed inputs but also computes and returns the log-Jacobian as the second
item in a two-item tuple. These modules are also equipped with a `reverse`
method that applies the inverse of the transformation.
"""
# pylint: disable=invalid-name

import copy
import io
import base64
from abc import abstractmethod, ABC
from typing import List
import torch
import numpy as np


__all__ = [
    "Module_",
    "ModuleList_",
    "MultiChannelModule_",
    "MultiOutChannelModule_",
    "PushforwardModule_",
    "InvisibilityMaskWrapperModule_",
]


# =============================================================================
class Module_(torch.nn.Module, ABC):
    """
    An abstract subclass of `torch.nn.Module` designed for creating invertible
    transformations that compute the logarithm of the Jacobian of the
    transformation.

    The trailing underscore in the class name indicates that the `forward`
    method not only returns the transformed inputs but also computes and
    returns the logarithm of the Jacobian determinant as the second item in a
    two-item tuple. This functionality is crucial for applications where the
    computation of the Jacobian is necessary, such as in normalizing flows.

    Transformations derived from this class are expected to be invertible.
    The `reverse` method applies the inverse of the transformation.

    To illustrate the use of this abstract class, consider the implementation
    of the hyperbolic tangent transformation using a subclass named `Tanh_`::


        class Tanh_(Module_):

            def forward(self, x, log0=0):
                '''
                Apply the hyperbolic tangent transformation.

                Parameters
                ----------
                x: torch.Tensor
                    Input tensor to be transformed.
                log0: float, optional
                    The logarithm of the Jacobian determinant from a previous
                    transformation. Default is 0.

                Returns
                -------
                y: torch.Tensor
                    Transformed output tensor after applying `tanh`.
                logj: float
                    Updated logarithm of the Jacobian determinant.
                '''
                y = torch.tanh(x)
                logj = -2 * torch.log(torch.cosh(x)).sum()
                return y, log0 + logj

            def reverse(self, y, log0=0):
                '''
                Apply the inverse hyperbolic tangent transformation.

                Parameters
                ----------
                y: torch.Tensor
                    Input tensor to be transformed.
                log0: float, optional
                    The logarithm of the Jacobian determinant from a previous
                    transformation. Default is 0.

                Returns
                -------
                x: torch.Tensor
                    Transformed output tensor after applying `atanh`.
                logj: float
                    Updated logarithm of the Jacobian determinant.
                '''
                x = torch.atanh(y)
                logj = 2 * torch.log(torch.cosh(x)).sum()
                return x, log0 + logj


    As the example shows, both the `forward` and `reverse` methods can accept
    an optional second input, `log0`, which allows users to carry over the
    logarithm of the Jacobian from a previous transformation. This feature
    makes it easy to chain multiple transformations together, ensuring that the
    logarithm of the Jacobian is computed cumulatively across all
    transformations.

    By inheriting from this class, users define their transformations with log
    Jacobian computations, streamlining the process of implementing complex
    probabilistic models.

    Note: The example provided does not consider a batch axis. It is
    recommended to include such a batch axis so that the log Jacobian is
    calculated for each sample separately, allowing for more efficient batch
    processing.
    """

    propagate_density = False

    def __init__(self, label=None):
        super().__init__()
        self.label = label

    @abstractmethod
    def forward(self, x, log0=0):
        """
        Perform the forward transformation.

        Args:
            x (Tensor): Input tensor to be transformed via the flow.
            log0 (Tensor | float, optional): Initial value for the log Jacobian
                from previous transformations. Defaults to 0.

        Returns:
            Tensor: Transformed output tensor.
            Tensor: Updated log Jacobian of the transformation.
        """

    @abstractmethod
    def reverse(self, x, log0=0):
        """
        Perform the reverse transformation.

        Args:
            x (Tensor): Input tensor to be transformed via the reverse flow.
            log0 (Tensor | float, optional): Initial value for the log Jacobian
                from previous transformations. Defaults to 0.

        Returns:
            Tensor: Transformed output tensor.
            Tensor: Updated log Jacobian of the reverse transformation.
        """

    def transfer(self, **kwargs):  # pylint: disable=unused-argument
        """Return a deep copy of this module."""
        return copy.deepcopy(self)

    @property
    def npar(self):
        """Return the total number of parameters."""
        return sum(np.prod(p.shape) for p in self.parameters())

    def sum_density(self, x: torch.Tensor):
        """Sum `x` over all but the batch axis, unless propagating density."""
        ndim = x.dim()
        if ndim < 2 or self.propagate_density:
            return x
        return torch.sum(x, dim=list(range(1, ndim)))

    def set_param2zero(self):
        """Zero out all parameters in place."""
        for param in self.parameters():
            torch.nn.init.zeros_(param)

    def get_weights_blob(self):
        """Return the state dict as a base64-encoded string."""
        serialized_model = io.BytesIO()
        torch.save(self.state_dict(), serialized_model)
        return base64.b64encode(serialized_model.getbuffer()).decode('utf-8')

    def set_weights_blob(self, blob, map_location=torch.device('cpu')):
        """Load parameters from a base64-encoded state dict blob."""
        weights = torch.load(
            io.BytesIO(base64.b64decode(blob.strip())),
            map_location=map_location,
            weights_only=True
        )
        self.load_state_dict(weights)

    def freeze_parameters(self):
        """Disable gradient tracking for all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        """Enable gradient tracking for all parameters."""
        for param in self.parameters():
            param.requires_grad = True


# =============================================================================
class ModuleList_(torch.nn.ModuleList, Module_):
    """
    A custom module that inherits from both `torch.nn.ModuleList` and `Module_`
    classes. This class is designed to manage a list of submodules that are
    themselves instances of `Module_`.

    By combining the functionalities of `torch.nn.ModuleList` and `Module_`,
    this class allows for efficient management of multiple invertible
    transformations, facilitating complex probabilistic modeling tasks.

    Note on `args`: `forward/reverse` accept an optional `args` for submodules
    that are conditioned on an externally supplied context. If provided, it is
    broadcast as-is to *every* submodule in the list -- so if you pass `args`,
    every submodule must accept an `args` parameter, whether or not it actually
    uses it. A `ModuleList_` mixing submodules that accept `args` with ones
    that don't must only be called with `args=None`; calling it with `args` set
    would raise a `TypeError` on the first submodule that doesn't accept it.
    """

    _groups = None

    def __init__(self, nets_: List[Module_]):
        super().__init__(nets_)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, log0=0, args=None):
        """
        Sequentially apply the forward transformations.

        Args:
            x (Tensor): Input to be transformed via the sequence of flows.
            log0 (Tensor | float, optional): Initial value for the log Jacobian
                from previous transformations. Defaults to 0.
            args (Tensor | tuple[Tensor, ...] | None, optional): Externally
                supplied conditioning input(s), forwarded unchanged to every
                submodule's `forward` if not None, and omitted otherwise (so
                submodules that don't accept `args` still work). Defaults to
                None.

        Returns:
            Tensor: Transformed output tensor after applying all flows.
            Tensor: Cumulative log Jacobians across flows.
        """
        logj = log0
        args_kwargs = {} if args is None else {'args': args}
        for net_ in self:
            x, logj = net_.forward(x, log0=logj, **args_kwargs)
        return x, logj

    def reverse(self, x, log0=0, args=None):
        """
        Sequentially apply the reverse transformations.

        Args:
            x (Tensor): Input to be transformed via the reverse sequence of
                reversed flows.
            log0 (Tensor | float, optional): Initial value for the log Jacobian
                from previous transformations. Defaults to 0.
            args (Tensor | tuple[Tensor, ...] | None, optional): Externally
                supplied conditioning input(s); must match the value used in
                the corresponding forward call. Forwarded unchanged to every
                submodule's `reverse` if not None, and omitted otherwise.
                Defaults to None.

        Returns:
            Tensor: Transformed output tensor after applying all reverse flows.
            Tensor: Cumulative log Jacobians across reverse flows.
        """
        logj = log0
        args_kwargs = {} if args is None else {'args': args}
        for net_ in reversed(self):
            x, logj = net_.reverse(x, log0=logj, **args_kwargs)
        return x, logj

    def grouped_parameters(self):
        """Return parameters, grouped by `_groups` if set, else a flat list."""
        if self._groups is None:
            return super().parameters()

        params_list = []

        def sum_list(x):
            return sum(x, start=[])

        for grp in self._groups:
            par = sum_list([list(self[k].parameters()) for k in grp['ind']])
            params_list.append({'params': par, **grp['hyper']})

        return params_list

    def setup_groups(self, groups=None):
        """If group is not None, it must be a list of dicts. e.g. as
        groups = [{'ind': [0, 1], 'hyper': dict(weight_decay=1e-4)},
                  {'ind': [2, 3], 'hyper': dict(weight_decay=1e-2)}]
        """
        self._groups = groups

    def hack(self, x, log0=0, args=None):
        """Similar to the forward method, except that returns the output of
        middle blocks too; useful for examining effects of each block.
        """
        stack = [(x, log0)]
        args_kwargs = {} if args is None else {'args': args}
        for net_ in self:
            x, log0 = net_.forward(x, log0, **args_kwargs)
            stack.append((x, log0))
        return stack

    def transfer(self, **kwargs):
        return self.__class__([net_.transfer(**kwargs) for net_ in self])

    def to(self, *args, **kwargs):
        for net_ in self:
            net_.to(*args, **kwargs)


# =============================================================================
class MultiChannelModule_(torch.nn.ModuleList):
    """
    Applies a separate network to each input channel.

    Similar to `Module_` but handles multiple channels individually. Each
    channel is processed by the corresponding network. Number of networks
    must match the number of channels.

    Args:
        nets_ (list[Module]): Networks, one per channel.
        channels_axis (int, default=1): Axis representing channels.
        keep_channels_axis (bool, default=True): Keep channel axis in output.
    """
    def __init__(
        self, nets_, channels_axis: int = 1, keep_channels_axis: bool = True
    ):
        super().__init__(nets_)
        self.channels_axis = channels_axis
        self.keep_channels_axis = keep_channels_axis

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, log0=0, args=None):
        """Apply each channel's network to the corresponding input."""
        return self._map(x, [n_.forward for n_ in self], log0=log0, args=args)

    def reverse(self, x, log0=0, args=None):
        """Apply each channel's network in reverse."""
        return self._map(x, [n_.reverse for n_ in self], log0=log0, args=args)

    def _map(self, x, f_, log0=0, args=None):
        """
        Apply each function in `f_` to corresponding element in input `x`.

        Splits input along channels_axis if keep_channels_axis, else unbinds.
        Recombines outputs and sums log-determinants.
        """
        # Split input into channels
        if self.keep_channels_axis:
            x = x.split(1, dim=self.channels_axis)
        else:
            x = x.unbind(dim=self.channels_axis)

        assert len(x) == len(f_), "Mismatch in channels and network."

        # Apply each function to its channel
        if args is None:
            out = [fj_(xj) for fj_, xj in zip(f_, x)]
        else:
            out = [fj_(xj, args=args) for fj_, xj in zip(f_, x)]

        # Recombine outputs
        if self.keep_channels_axis:
            x = torch.cat([o[0] for o in out], dim=self.channels_axis)
        else:
            x = torch.stack([o[0] for o in out], dim=self.channels_axis)

        # Sum log-determinants
        logj = sum(o[1] for o in out)

        return x, log0 + logj

    @property
    def npar(self):
        """Return the total number of parameters."""
        return sum(np.prod(p.shape) for p in self.parameters())


# =============================================================================
class MultiOutChannelModule_(MultiChannelModule_):
    """
    Applies multiple networks to the entire input, then concatenates outputs.

    Unlike `MultiChannelModule_`, each network sees the full input rather than
    a single channel.
    """

    def _map(self, x, f_, log0=0, args=None):
        """
        Apply each function in `f_` to the input `x` and concatenate outputs.
        """
        # Apply each function to the full input
        if args is None:
            out = [fj_(x) for fj_ in f_]
        else:
            out = [fj_(x, args=args) for fj_ in f_]

        # Concatenate outputs along channels axis
        x = torch.cat([o[0] for o in out], dim=self.channels_axis)

        # Sum log-determinants
        logj = sum(o[1] for o in out)

        return x, log0 + logj


# =============================================================================
class PushforwardModule_(Module_):
    """
    Applies `transform_` not to `x` directly, but to a derived object `y` built
    from `x` and `args` via `pushforward`, then writes the transformed `y` back
    onto `x` via `pullback`.

    Parameters
    ----------
    pushforward : Callable[[x, args], y]
    transform_ : Module_, which is applied to `y` alone.
    pullback : Callable[[y, args], x]
    """

    def __init__(self, pushforward, transform_, pullback):
        super().__init__()
        self.pushforward = pushforward
        self.transform_ = transform_
        self.pullback = pullback

    def forward(self, x, log0=0, args=None):
        y = self.pushforward(x, args)
        y, logj = self.transform_.forward(y)
        x = self.pullback(y, args)
        return x, log0 + logj

    def reverse(self, x, log0=0, args=None):
        y = self.pushforward(x, args)
        y, logj = self.transform_.reverse(y)
        x = self.pullback(y, args)
        return x, log0 + logj


# =============================================================================
class InvisibilityMaskWrapperModule_(Module_):
    """
    A wrapper that makes a part of the input invisible before passing it the
    underlying network (`net_`).

    Parameters
    ----------
    net_ : instance of Module_
        should not have any other nested net_ that keeps track of Jacobian
        of transformation.

    mask : instance of Mask
        for partitioning the input data to visible and invisible parts.
    """

    def __init__(self, net_, *, mask):
        super().__init__()
        self.net_ = net_
        self.mask = mask
        self.net_.propagate_density = True  # does not sum the density

    def forward(self, x, log0=0, args=None):
        x_v, x_invisible = self.mask.split(x)  # x_v: x_visible
        if args is None:
            x_v, logj_density = self.net_.forward(x_v)
        else:
            x_v, logj_density = self.net_.forward(x_v, args=args)
        x_v = self.mask.purify(x_v, channel=0)
        logj = self.sum_density(self.mask.purify(logj_density, channel=0))
        return self.mask.cat(x_v, x_invisible), log0 + logj

    def reverse(self, x, log0=0, args=None):
        x_v, x_invisible = self.mask.split(x)  # x_v: x_visible
        if args is None:
            x_v, logj_density = self.net_.reverse(x_v)
        else:
            x_v, logj_density = self.net_.reverse(x_v, args=args)
        x_v = self.mask.purify(x_v, channel=0)
        logj = self.sum_density(self.mask.purify(logj_density, channel=0))
        return self.mask.cat(x_v, x_invisible), log0 + logj
