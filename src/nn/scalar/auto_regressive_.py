# Copyright (c) 2024 Javad Komijani

import torch
import numpy as np

from normflow.nn import Module_
from normflow.nn import ModuleList_

from abc import abstractmethod, ABC

from typing import List

Tensor = torch.Tensor  # for typing


# =============================================================================
class ARMask(ABC):
    """An abstract base class for masks suitable for auto-regressive models."""

    @abstractmethod
    def split(self, var: Tensor) -> List[Tensor]:
        """Splits the input, suitable for the forward process."""
        pass

    @abstractmethod
    def cat(self, var_list: List[Tensor]) -> Tensor:
        """Reverses the `split` method, suitable for the reverse process."""
        pass

    @abstractmethod
    def stepwise_cat(
        self, frozen_var: Tensor, active_var: Tensor, ind: int
        ) -> Tensor:
        """
        Concatenates `frozen_var` and `active_var` in order to create an
        updated frozen variable for the next layer, suitable for the forward
        process.

        The inputs are defined as::

            active_var[n] = split(var)[n]
            frozen_var[n] = stepwise_cat(frozen_var[n - 1], active_var[n - 1])

        with `frozen_var[0] = None`.
        """
        pass

    @abstractmethod
    def stepwise_split(self, var: Tensor, ind: int) -> [Tensor, Tensor]:
        """
        Reverses the `stepwise_cat` method, i.e., splits the input to active
        and frozen parts, suitable for the reverse process.
        """
        pass


# =============================================================================
class ARModule_(ModuleList_):
    """
    This module is a base module to implement autoregressive (AR) models,
    a type of model where each prediction in a sequence is conditioned on
    previous outputs. Mathematically, this can be represented as::

        P(x_n | x_{n-1}, x_{n-2}, ..., x_0)

    where the probability of the current output ``x_n`` is conditioned on the
    previous outputs in the sequence ``x_{n-1}, x_{n-2}, ..., x_0``.

    To instantiate it, one should provide a valid partitioner (mask) to split
    the data into `N` parts and concatenate the parts to make a whole.
    Moreover, one should provide appropriate `N` sub-modules to transform each
    part conditioned to the previous parts. The first submodule takes the first
    items of the splitted data, `x_0`, as an active variable, updates it, and
    returns the updated `x_0` along with the Jacobian of transformation.
    The updated `x_0` then will be considered the frozen variable for the next
    submodule, which updates `x_1` conditioned to `x_0`. A mixture of updated
    `x_0` and `x_1` will server the role of the frozen variable for the next
    layer and so on. In summary, except for the first submodule, each submodule
    takes the n-th part of the splitted data as an active variable and a Tensor
    of all previous parts as a frozen variable. The submodule transforms the
    active variables, using a fucntion that depends on the frozen variables,
    and returns a tuple, which contains both updated active variables and
    frozen varaibles, and the Jacobian of the transformation.
    """

    def __init__(self, nets_: List[Module_], mask: ARMask):

        super().__init__(nets_)

        self.mask = mask
        self.n_layers = len(nets_)

    def forward(self, var, log0=0):

        var_list = self.mask.split(var)

        for ind, subnet_ in enumerate(self):
            actv_var = var_list[ind]
            if ind == 0:
                actv_var, log0 = subnet_(actv_var, log0=log0)
                frzn_var = actv_var  # for the next layer
            else:
                (actv_var, _), log0 = subnet_([actv_var, frzn_var], log0=log0)
                frzn_var = self.mask.stepwise_cat(frzn_var, actv_var, ind)

        return frzn_var, log0

    def reverse(self, var, log0=0):

        var_list = [None] * self.n_layers

        for ind in range(self.n_layers - 1, -1, -1):
            # f_var: frozen variable
            # a_var: active variable
            f_var, a_var = self.mask.stepwise_split(var, ind)
            if ind > 0:
                (a_var, _), log0 = self[ind].reverse([a_var, f_var], log0=log0)
            else:
                a_var, log0 = self[ind].reverse(a_var, log0=log0)
            var_list[ind] = a_var
            var = f_var  # for the next round

        return self.mask.cat(var_list), log0

    def hack(self, var, log0=0):
        pass


class VectorARModule_(ARModule_):

    def __init__(self, nets_: List[Module_]):

        super().__init__(nets_, VectorARMask())


class FiboARModule_(ARModule_):

    def __init__(self, nets_: List[Module_], shape: List[int]):

        super().__init__(nets_, FiboARMask(shape))


# =============================================================================
class VectorARMask(ARMask):
    """A sublcass of `ARMask` suitable for small 1d tensors."""

    def split(self, var):
        return torch.split(var, 1, dim=-1)

    def cat(self, var_list):
        return torch.cat(var_list, dim=-1)

    def stepwise_cat(self, frozen_var, active_var, ind):  # ind is irrelavant
        return torch.cat([frozen_var, active_var], dim=-1)

    def stepwise_split(self, var, ind):  # ind is irrelavant
        return torch.split(var, [var.shape[-1] - 1, 1], dim=-1)


class FiboARMask(ARMask):
    """A sublcass of `ARMask` suitable for large lattices because of its
    exponential decrease in number of layers. It is called `Fibo` due to its
    pattern of splitting of the data that is similar to Fibonacci tiling.

    To instantiate the class, one should provide a tuple/list of integers as
    the shape of the lattice.

    The input variable is splitted using a set of ``slices``, where the first
    slice returns 1 d.o.f., and the other slices return 1, 2, 4, 8, 16 dofs and
    so on, respectively, if the length of the lattice is a power of 2.
    Otherwise, the other slice sizes would not follow the exact geometric
    pattern.

    Let us consider `shape = [8, 8, 8]`, the tesosr is divided into sub-tensors
    of shape::

        [4, 8, 8]
        [4, 4, 8]
        [4, 4, 4]
        [2, 4, 4]
        [2, 2, 4]
        [2, 2, 2]
        [1, 2, 2]
        [1, 1, 2]
        [1, 1, 1]
        [1, 1, 1]

    and the `split` method returns the slices from bottom to top.
    Another example is for `shape = [7, 8]`, where the tensor is divided into
    sub-tensors of shape::

        [3, 8]
        [4, 4]
        [2, 4]
        [2, 2]
        [1, 2]
        [1, 1]
        [1, 1]
    """

    metadata_labels = \
            ('shape', 'bite_axis', 'bite_len', 'bite_shape', 'leftover_shape')

    def __init__(self, shape: List[int]):

        slices_list = []
        metadata_list = []

        axis, leftovoer_shape = 0, shape

        while not (leftovoer_shape is None):
            out = self.bite_along_axis(shape, axis)
            slice_list, bite_length, bite_shape, leftovoer_shape = out
            if not (bite_length is None):
                slices_list.append(slice_list)
                metadata_list.append((shape, axis, *out[1:]))
            axis = (axis + 1) % len(shape)  # increase for the next round
            shape = leftovoer_shape

        self.slices_list = slices_list[::-1]
        self.metadata_list = metadata_list[::-1]

    def __repr__(self):

        labels = ', '.join(self.metadata_labels)
        metdata = '\n'.join([f"{val}\t" for val in self.metadata_list[::-1]])

        return f"Fibonacci Auto-Regressive Mask\n{labels}\n{metdata}"

    def split(self, var: Tensor) -> List[Tensor]:
        """Splits the input, suitable for the forward process."""
        return [var[slices] for slices in self.slices_list]

    def cat(self, var_list: List[Tensor]) -> Tensor:
        """Reverses the `split` method, suitable for the reverse process."""
        item = var_list[-1]
        shape = (item.shape[0], *self.metadata_list[-1][0])
        var = torch.zeros(shape, dtype=item.dtype, device=item.device)
        for item, slices in zip(var_list, self.slices_list):
            var[slices] = item
        return var

    def stepwise_cat(
        self, frozen_var: Tensor, active_var: Tensor, ind: int
        ) -> Tensor:
        """Concatenates `frozen_var` and `active_var` in order to create
        an updated frozen variable for the next layer, suitable for the forward
        process.

        The inputs are defined as::

            active_var[n] = split(var)[n]
            frozen_var[n] = stepwise_cat(frozen_var[n - 1], active_var[n - 1])

        with `frozen_var[0] = None`.
        """
        if ind == 0:
            return active_var

        lat_shape, axis = self.metadata_list[ind][:2]
        pre_axes = [slice(None)] * (1 + axis)  # 1 for the batch axis
        shape = (active_var.shape[0], *lat_shape)

        var = torch.zeros(shape, dtype=active_var.dtype, device=active_var.device)
        var[pre_axes + [slice(0, None, 2)]] = frozen_var
        var[pre_axes + [slice(1, None, 2)]] = active_var

        return var

    def stepwise_split(self, var: Tensor, ind: int) -> [Tensor, Tensor]:
        """Reverses the `stepwise_cat` method, i.e., splits the input to active
        and frozen parts, suitable for the reverse process.
        """

        if ind == 0:
            return None, var

        _, axis = self.metadata_list[ind][:2]
        pre_axes = [slice(None)] * (1 + axis)    # 1 for the batch axis

        frozen_var = var[pre_axes + [slice(0, None, 2)]]
        active_var = var[pre_axes + [slice(1, None, 2)]]

        return frozen_var, active_var

    @staticmethod
    def bite_along_axis(shape, axis):
        """Returns a list of `slices` for "taking a bite" and returns the shape
        of the leftover piece. It is assumed `shape` specifies the shape of the
        lattice and the bite is along `axis` axis excluding the batch axis.
        """

        for_batch_axis = [slice(0, None)]

        if np.prod(shape) == 1:
            bite_slice_list = for_batch_axis + [slice(0, 1) for _ in shape]
            bite_length = 1
            bite_shape = shape
            leftover_shape = None  # a flag to signal the end of the procedure

        elif shape[axis] == 1:
            bite_slice_list = None  # cannot split along the `axis` diection
            bite_length = None
            bite_shape = None
            leftover_shape = shape

        else:
            # split the lattice along the `axis` direction
            bite_slice_list = for_batch_axis + [slice(0, ell) for ell in shape]
            ell = shape[axis]
            bite_slice_list[1 + axis] = slice((1 + ell) // 2, ell)
            bite_length = ell // 2
            bite_shape = list(shape)
            bite_shape[axis] = ell // 2
            leftover_shape = list(shape)
            leftover_shape[axis] = (1 + ell) // 2

        return bite_slice_list, bite_length, bite_shape, leftover_shape


# =============================================================================
def test_fibo_ar_mask(shape=[4, 13]):
    mask = FiboARMask(shape)
    print(mask)

    x = torch.randn(2, *shape)

    print("\nChecking if mask.cat is inverse of mask.split:""")
    var_list = mask.split(x)
    print(mask.cat(var_list) / x)

    print("\nChecking if stepwise_split is inverse of stepwise_cat:""")
    a, b = var_list[:2]
    y = mask.stepwise_cat(a, b, 1)
    c, d = mask.stepwise_split(y, 1)
    print((c / a).ravel())
    print((d / b).ravel())
