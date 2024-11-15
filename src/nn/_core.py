# Copyright (c) 2021-2024 Javad Komijani


import torch
import numpy as np
import copy
import io
import base64
from abc import abstractmethod, ABC
from typing import Type, List


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
        pass

    @abstractmethod
    def reverse(self, x, log0=0):
        pass

    def transfer(self, **kwargs):
        return copy.deepcopy(self)

    @property
    def npar(self):
        return sum([np.prod(p.shape) for p in self.parameters()])

    def sum_density(self, x):
        if self.propagate_density:
            return x
        else:
            return torch.sum(x, dim=list(range(1, x.dim())))

    def set_param2zero(self):
        for param in self.parameters():
            torch.nn.init.zeros_(param)

    def get_weights_blob(self):
        serialized_model = io.BytesIO()
        torch.save(self.state_dict(), serialized_model)
        return base64.b64encode(serialized_model.getbuffer()).decode('utf-8')

    def set_weights_blob(self, blob, map_location=torch.device('cpu')):
        weights = torch.load(
                io.BytesIO(base64.b64decode(blob.strip())),
                map_location=map_location,
                weights_only=True
                )
        self.load_state_dict(weights)

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
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
    """

    _groups = None

    def __init__(self, nets_: List[Module_]):
        super().__init__(nets_)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, log0=0):
        for net_ in self:
            x, log0 = net_.forward(x, log0=log0)
        return x, log0

    def reverse(self, x, log0=0):
        for net_ in list(self)[::-1]:  # list() is needed for child classes...
            x, log0 = net_.reverse(x, log0=log0)
        return x, log0

    def grouped_parameters(self):
        if self._groups is None:
            return super().parameters()
        else:
            params_list = []
            sum_ = lambda x: sum(x, start=[])
            for grp in self._groups:
                par = sum_([list(self[k].parameters()) for k in grp['ind']])
                params_list.append(dict(params=par, **grp['hyper']))
            return params_list

    def setup_groups(self, groups=None):
        """If group is not None, it must be a list of dicts. e.g. as
        groups = [{'ind': [0, 1], 'hyper': dict(weight_decay=1e-4)},
                  {'ind': [2, 3], 'hyper': dict(weight_decay=1e-2)}]
        """
        self._groups = groups

    def hack(self, x, log0=0):
        """Similar to the forward method, except that returns the output of
        middle blocks too; useful for examining effects of each block.
        """
        stack = [(x, log0)]
        for net_ in self:
            x, log0 = net_.forward(x, log0)
            stack.append((x, log0))
        return stack

    def transfer(self, **kwargs):
        return self.__class__([net_.transfer(**kwargs) for net_ in self])

    def to(self, *args, **kwargs):
        for net_ in self:
            net_.to(*args, **kwargs)


# =============================================================================
class MultiChannelModule_(torch.nn.ModuleList):
    """A prototype class similar to `Module_` except that it handles multiple
    channels seperately, in the sense that each channel is transformed by
    corresponding NN. The number of input NNs must agree with the number of
    channels.
    """

    def __init__(self, nets_,
            label=None, channels_axis=1, keep_channels_axis=True):
        super().__init__(nets_)
        self.channels_axis = channels_axis
        self.keep_channels_axis = keep_channels_axis
        self.label = label

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, log0=0):
        return self._map(x, [net_.forward for net_ in self], log0=log0)

    def reverse(self, x, log0=0):
        return self._map(x, [net_.reverse for net_ in self], log0=log0)

    def _map(self, x, f_, log0=0):
        if self.keep_channels_axis:
            x = x.split(1, dim=self.channels_axis)
        else:
            x = x.unbind(dim=self.channels_axis)

        assert len(x) == len(f_), "mismatch in channels of input & network."

        out = [fj_(xj) for fj_, xj in zip(f_, x)]
        if self.keep_channels_axis:
            x = torch.cat([o[0] for o in out], dim=self.channels_axis)
        else:
            x = torch.stack([o[0] for o in out], dim=self.channels_axis)
        logJ = sum([o[1] for o in out])

        return x, log0 + logJ

    def parameters(self):
        return super().parameters()

    @property
    def npar(self):
        return sum([np.prod(p.shape) for p in super().parameters()])


# =============================================================================
class MultiOutChannelModule_(MultiChannelModule_):

    def _map(self, x, f_, log0=0):

        out = [fj_(x) for fj_ in f_]
        x = torch.cat([o[0] for o in out], dim=self.channels_axis)
        logJ = sum([o[1] for o in out])

        return x, log0 + logJ


# =============================================================================
class InvisibilityMaskWrapperModule_(Module_):
    """A wrapper that makes a part of the input invisible before passing it the
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
        super().__init__(label=f'wrapper:{net_.label}')
        self.net_ = net_
        self.mask = mask
        self.net_.propagate_density = True  # does not sum the density

    def forward(self, x, log0=0):
        x_v, x_invisible = self.mask.split(x)  # x_v: x_visible
        x_v, logJ_density = self.net_.forward(x_v)
        x_v = self.mask.purify(x_v, channel=0)
        logJ = self.sum_density(self.mask.purify(logJ_density, channel=0))
        return self.mask.cat(x_v, x_invisible), log0 + logJ

    def reverse(self, x, log0=0):
        x_v, x_invisible = self.mask.split(x)  # x_v: x_visible
        x_v, logJ_density = self.net_.reverse(x_v)
        x_v = self.mask.purify(x_v, channel=0)
        logJ = self.sum_density(self.mask.purify(logJ_density, channel=0))
        return self.mask.cat(x_v, x_invisible), log0 + logJ
