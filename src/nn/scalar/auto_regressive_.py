# Copyright (c) 2024 Javad Komijani

import torch
import numpy as np

from .._core import Module_, ModuleList_
from .modules import ConvAct
from .modules_ import DistConvertor_
from .couplings_ import AffineCoupling_
from ...mask import EvenOddMask
from ...mask.partitioner import ListPartitioner


class LatticeAutoReg_(ModuleList_):
    """An Auto-regressive model with exponential improvement in number of
    layers suitable for large lattices.

    Parameters
    ----------
    lat_shape: Tuple[int, ...]
        shape of underlying lattice
    nets_: Union[Tuple[Module_, ...], None]
        a tuple of instances of `Module_` or None. Default is None, indicating
        a default set of `Modules` that are based on `DistConvetor_` and
        `AffineCoupling_` are used.
    """

    def __init__(self, lat_shape, nets_=None):

        if nets_ is None:
            nets_ = LatticeAutoReg_.make_auto_regression_modules(lat_shape)

        super().__init__(nets_)

        self.all_slices = self.get_slices(lat_shape)[::-1]
        self.n_depth = len(self.all_slices)

    def forward(self, var, log0=0):
        var_list = self.split(var)
        frozen_var = None
        for ind in range(self.n_depth):
            subnet_ = self[ind]
            mixed_var, log0 = subnet_(var_list[ind], frozen_var, log0=log0)
            frozen_var = mixed_var  # for the next round
        return mixed_var, log0

    def reverse(self, mixed_var, log0=0):
        var_list = [None] * len(self.all_slices)
        for ind in range(self.n_depth - 1, -1, -1):
            subnet_ = self[ind]
            var, frozen_var, log0 = subnet_.reverse(mixed_var, log0=log0)
            var_list[ind] = var
            mixed_var = frozen_var  # for the next round
        return self.cat(var_list), log0

    def hack(self, var, log0=0):
        var_list = self.split(var)
        frozen_var = None
        stack = []
        for ind in range(self.n_depth):
            subnet_ = self[ind]
            mixed_var, log0 = subnet_(var_list[ind], frozen_var, log0=log0)
            stack.append((mixed_var, log0))
            frozen_var = mixed_var  # for the next round
        return stack

    def split(self, var):
        return [var[slices] for slices in self.all_slices]

    def cat(self, splitted_list):
        item = splitted_list[-1]
        lat_shape = [l * (2 if k==1 else 1) for k, l in enumerate(item.shape)]
        var = torch.zeros(lat_shape, dtype=item.dtype, device=item.device)
        for item, slices in zip(splitted_list, self.all_slices):
            var[slices] = item
        return var

    @classmethod
    def get_slices(cls, shape, axis=0):
        assert shape[axis] % 2 == 0 or shape[axis] == 1, "OOPS: must be 2^n"
        shape = list(shape)
        if np.prod(shape) == 1:
            return [[slice(0, None)] + [slice(0, 1) for _ in shape]]
        ell = shape[axis]
        if ell > 1:
            split_list = [[slice(0, None)] + [slice(0, ell) for ell in shape]]
            split_list[0][1 + axis] = slice(ell // 2, ell)
            shape[axis] = ell // 2
        else:
            split_list = []
        return split_list + cls.get_slices(shape, (1 + axis) % len(shape))

    @classmethod
    def get_axes_shape(cls, shape, axis=0):
        shape = list(shape)
        if np.prod(shape) == 1:
            return [(axis, shape)]
        ell = shape[axis]
        if ell > 1:
            shape[axis] = ell // 2
            split_list = [(axis, shape)]
        else:
            split_list = []
        return split_list + cls.get_axes_shape(shape, (1 + axis) % len(shape))

    @classmethod
    def make_auto_regression_modules(
            cls, lat_shape, knots_len=50, hidden_sizes=[8, 8]
            ):
        all_axes_shape = cls.get_axes_shape(lat_shape)[::-1]
        nets_ = [None] * len(all_axes_shape)
        conv_dict = dict(
            in_channels=1,
            out_channels=2,
            kernel_size=3,
            conv_dim=len(lat_shape),
            hidden_sizes=hidden_sizes,
            acts=(*['leaky_relu']*len(hidden_sizes), None)
            )
        for k, (axis, shape) in enumerate(all_axes_shape):
            if k == 0:
                nets_[0] = AutoRegWrapper4DistConvertor_(knots_len)
            else:
                nets1 = [ConvAct(**conv_dict) for _ in range(4)]
                nets2 = [ConvAct(**conv_dict) for _ in range(4)]
                nets_[k] = AutoRegSubmodule_(nets1, nets2, shape, expansion_axis=axis)
        return nets_


class AutoRegSubmodule_(Module_):
    """A submodule for Auto-regressive model with exponential improvement in
    number of layers suitable for large lattices.

    The submodule takes active and frozen variables, transforms the active
    variables, concatenates the transformed and frozen variables, and returns
    them, which can be used as a frozen variable for the next submodule of the
    autoregressive model.

    Parameters
    ----------
    lat_shape: Tuple[int, ...]
        shape of underlying lattice
    nets1: Tuple[Module, ...]
        a tuple of instances of `Module` that are used in constrcuting an
        instance of `AffineCoupling_` that takes as input both active and
        frozen variables.
    nets2: Tuple[Module, ...]
        a tuple of instances of `Module` that are used in constrcuting an
        instance of `AffineCoupling_` that takes as input the transformed
        variables.
    knots_len: int (optional)
        specifies the number of knots in `DistConvertor_` that acts on the
        trasformed data.
    expansion_axis: int (optional)
        the axis for concatenating the transformed and frozen variables.
        Default is 0, excluding the batch axis.
    """

    def __init__(self, nets1, nets2, shape, expansion_axis=0, knots_len=10):
        super().__init__()
        self.expansion_axis = expansion_axis  # excluding the batch axis
        self.dc_ = DistConvertor_(knots_len, smooth=True)
        self.affine1_ = ModuleList_(
            [AffineCoupling_([net], mask=ListPartitioner()) for net in nets1]
            )
        self.affine2_ = AffineCoupling_(nets2, mask=EvenOddMask(shape=shape))

    def forward(self, var, frozen_var, log0=0):
        (var, _), logj = self.affine1_([var, frozen_var], log0=log0)
        var, logj = self.affine2_(var, log0=logj)
        var, logj = self.dc_(var, log0=logj)
        mixed_var = self.mixer(var, frozen_var)
        return mixed_var, logj

    def reverse(self, mixed_var, log0=0):
        var, frozen_var = self.reverse_mixer(mixed_var)
        var, logj = self.dc_.reverse(var, log0=log0)
        var, logj = self.affine2_.reverse(var, log0=logj)
        (var, _), logj = self.affine1_.reverse([var, frozen_var], log0=logj)
        return var, frozen_var, logj

    def mixer(self, var, frozen_var):
        axis = 1 + self.expansion_axis  # 1 for the batch axis
        shape = list(var.shape)
        shape[axis] *= 2
        pre_axes = [slice(None)] * axis

        mixed_var = torch.zeros(shape, dtype=var.dtype, device=var.device)
        # mixed_var[pre_axes + [slice(0, None, 2)]] = (frozen_var + var / 2)
        # mixed_var[pre_axes + [slice(1, None, 2)]] = (frozen_var - var / 2)
        mixed_var[pre_axes + [slice(0, None, 2)]] = var
        mixed_var[pre_axes + [slice(1, None, 2)]] = frozen_var
        return mixed_var

    def reverse_mixer(self, mixed_var):
        axis = 1 + self.expansion_axis  # 1 for the batch axis
        pre_axes = [slice(None)] * axis
        a = mixed_var[pre_axes + [slice(0, None, 2)]]
        b = mixed_var[pre_axes + [slice(1, None, 2)]]
        return a, b
        # return (a - b), (a + b) / 2


class AutoRegWrapper4DistConvertor_(DistConvertor_):

    def forward(self, var, frozen_var, log0=0):
        return super().forward(var, log0=log0)

    def reverse(self, var, log0=0):
        var, log0 = super().reverse(var, log0=log0)
        return var, None, log0
