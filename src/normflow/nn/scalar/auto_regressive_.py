# Copyright (c) 2024-2025 Javad Komijani

"""
This is a module for auto-regressive models
"""

# pylint: disable=invalid-name

from abc import abstractmethod, ABC
from typing import List

import torch
import numpy as np

from normflow.nn import Module_
from normflow.nn import ModuleList_

Tensor = torch.Tensor  # for typing

__all__ = ["AutoRegModule_", "FibonacciTilingSplitter", "VectorAutoRegModule_"]


# =============================================================================
class AutoRegSplitter(ABC):
    """An abstract base class for splitting data for auto-regressive models."""

    @abstractmethod
    def split(self, x: Tensor) -> List[Tensor]:
        """Splits the input, suitable for the forward process."""

    @abstractmethod
    def cat(self, x_list: List[Tensor]) -> Tensor:
        """Reverses the `split` method, suitable for the reverse process."""

    @abstractmethod
    def stepwise_cat(
        self, x_frozen: Tensor, x_active: Tensor, ind: int
    ) -> Tensor:
        """
        Concatenates `x_frozen` and `x_active` in order to create an
        updated frozen variable for the next layer, suitable for the forward
        process.

        The inputs are defined as::

            x_active[n] = split(x)[n]
            x_frozen[n] = stepwise_cat(x_frozen[n - 1], x_active[n - 1])

        with `x_frozen[0] = None`.
        """

    @abstractmethod
    def stepwise_split(self, x: Tensor, ind: int) -> [Tensor, Tensor]:
        """
        Reverses the `stepwise_cat` method, i.e., splits the input to active
        and frozen parts, suitable for the reverse process.
        """


# =============================================================================
class AutoRegModule_(ModuleList_):
    """
    Base class for implementing autoregressive (AR) models.

    In an AR model, each variable is transformed (or predicted) conditioned
    on all previously processed variables. Formally::

        P(x_n | x_{n-1}, x_{n-2}, ..., x_0)

    where the probability of the current variable `x_n` depends on all
    earlier variables in the sequence.

    Parameters
    ----------
    nets_ : List[Module_]
        A list of `N` submodules. Each submodule transforms one part of the
        input while conditioning on the previously updated parts.
    splitter : AutoRegSplitter
        A splitter that divides the input tensor into `N` parts and
        reassembles them after transformation.

    How it works
    ------------
    1. The splitter divides the input tensor into `N` parts:
       `x_0, x_1, ..., x_{N-1}`. Each part may represent multiple degrees
       of freedom.
    2. The model applies `N` submodules sequentially:
       - **Submodule 0**:
         - Takes `x_0` as the active variable,
         - Transforms it directly (no conditioning),
         - Produces the first frozen variable for subsequent layers.
       - **Submodule n > 0**:
         - Takes `x_n` as the active variable,
         - Conditions its transformation on the frozen variables
           (the concatenation of all previously updated parts),
         - Produces an updated active variable and extends the frozen state.
    3. The forward pass updates parts in order (`x_0` → `x_{N-1}`),
       while the reverse pass reconstructs them in reverse.

    Implementation details
    ----------------------
    - During the forward pass, the frozen state grows at each step via
      `splitter.stepwise_cat`.
    - During the reverse pass, `splitter.stepwise_split` separates active
      and frozen variables.
    - Each submodule also contributes to the accumulated log-determinant
      of the Jacobian, ensuring invertibility and proper density estimation.

    Returns
    -------
    Tuple[Tensor, Tensor]
        - The transformed (or reconstructed) tensor,
        - The accumulated log-determinant of the Jacobians.
    """

    def __init__(self, nets_: List[Module_], splitter: AutoRegSplitter):

        super().__init__(nets_)

        self.splitter = splitter
        self.n_layers = len(nets_)

    def forward(self, x, log0=0):

        x_list = self.splitter.split(x)

        for ind in range(self.n_layers):
            forward = self[ind].forward
            x_active = x_list[ind]
            if ind == 0:
                x_active, logj = forward(x_active, log0=log0)
                x_frozen = x_active  # for the next layer
            else:
                (x_active, _), logj = forward((x_active, x_frozen), log0=logj)
                x_frozen = self.splitter.stepwise_cat(x_frozen, x_active, ind)

        return x_frozen, logj

    def reverse(self, x, log0=0):

        x_frozen = x
        x_list = [None] * self.n_layers

        for ind in range(self.n_layers - 1, -1, -1):
            reverse = self[ind].reverse
            x_frozen, x_active = self.splitter.stepwise_split(x_frozen, ind)
            if ind > 0:
                (x_active, _), log0 = reverse((x_active, x_frozen), log0=log0)
            else:
                x_active, log0 = self[ind].reverse(x_active, log0=log0)
            x_list[ind] = x_active

        return self.splitter.cat(x_list), log0


class VectorAutoRegModule_(AutoRegModule_):
    """A sublcass of `AutoRegModule_` suitable for small 1-dim tensors."""

    def __init__(self, nets_: List[Module_]):
        super().__init__(nets_, VectorAutoRegSplitter())


class FibonacciAutoRegModule_(AutoRegModule_):
    """
    Autoregressive module specialized for high-dimensional lattices.

    This class is a wrapper around `AutoRegModule_` that pairs it with
    a `FibonacciTilingSplitter`. The splitter partitions a lattice into
    progressively larger regions according to a Fibonacci-like tiling pattern,
    which reduces the number of autoregressive layers required for large
    inputs.

    Notes
    -----
    - This class serves mainly as a convenience template.
    - Users may directly instantiate `AutoRegModule_` with
      `FibonacciTilingSplitter`.
    """

    def __init__(self, nets_: List[Module_], shape: List[int]):
        super().__init__(nets_, FibonacciTilingSplitter(shape))


# =============================================================================
class VectorAutoRegSplitter(AutoRegSplitter):
    """A sublcass of `AutoRegSplitter` suitable for small 1d tensors."""

    def split(self, x):
        return torch.split(x, 1, dim=-1)

    def cat(self, x_list):
        return torch.cat(x_list, dim=-1)

    def stepwise_cat(self, x_frozen, x_active, ind):  # ind is irrelavant
        return torch.cat([x_frozen, x_active], dim=-1)

    def stepwise_split(self, x, ind):  # ind is irrelavant
        return torch.split(x, [x.shape[-1] - 1, 1], dim=-1)


class FibonacciTilingSplitter(AutoRegSplitter):
    """
    Splits lattices of any dimensionality efficiently using a hierarchical
    pattern inspired by Fibonacci tiling.

    This subclass of `AutoRegSplitter` extends the concept of Fibonacci tiling,
    which traditionally works in two dimensions, to lattices of arbitrary
    dimensionality. This splitter reduces the number of layers exponentially
    compared to sequential splitting strategies, making it suitable for
    high-dimensional lattices and maintaing multiscale decomposition.

    Parameters
    ----------
    shape : tuple or list of int
        The shape of the lattice to split. Each dimension specifies the size
        along that axis.

    Behavior
    --------
    The input tensor is split into subtensors using a series of slices:
    - The first slice returns 1 degree of freedom (d.o.f.).
    - Subsequent slices increase in size geometrically (1, 2, 4, 8, ...),
      if the lattice length along a given axis is a power of 2.
    - For lattice lengths not equal to a power of 2, the slices have a similar
      hierarchical pattern but do not strictly follow geometric progression.

    - Example: consider `shape = [8, 8, 8]`, the tesosr is divided into
      sub-tensors of shape::

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

       The `split` method returns the slices from bottom to top.

    - Exampel: consider `shape = [7, 8]`, where the tensor is divided into
      sub-tensors of shape::

        [3, 8]
        [4, 4]
        [2, 4]
        [2, 2]
        [1, 2]
        [1, 1]
        [1, 1]

    Metadata and Slices
    -------------------
    The class stores two main lists for managing splits and recombination:

    1. `metadata_list`:
        A list of tuples, where each tuple corresponds to one split ("bite") of
        the lattice:

            (shape, bite_axis, bite_len, bite_shape, leftover_shape)

        - `shape`: The shape of the lattice **before this bite/split**.
        - `bite_axis`: The axis along which the bite/split is performed.
        - `bite_len`: The length of the bite/slice along that axis.
        - `bite_shape`: The shape of the subtensor returned by this bite/split.
        - `leftover_shape`: The shape of the remaining lattice after this
           bite/split (or `None` if the bite/split is the last one).

        This metadata is used to:
        - Reconstruct the full tensor in `cat` and `stepwise_cat`.
        - Provide insight into how the lattice is recursively divided.

    2. `slices_list`:
        A list of Python slice objects (or tuples of slices) corresponding to
        each bite/split. Each element specifies exactly which part of the input
        tensor to extract for that split. This list is used directly in the
        `split` method to slice the input tensor and in `cat` / `stepwise_cat`
        to recombine subtensors in the correct positions.
    """

    metadata_labels = (
        'shape', 'bite_axis', 'bite_len', 'bite_shape', 'leftover_shape'
    )

    def __init__(self, shape: List[int]):

        slices_list = []
        metadata_list = []

        axis, leftovoer_shape = 0, shape

        while leftovoer_shape is not None:
            out = self.bite_along_axis(shape, axis)
            slice_list, bite_length, _, leftovoer_shape = out
            if bite_length is not None:
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

    def split(self, x: Tensor) -> List[Tensor]:
        """Splits the input, suitable for the forward process."""
        return [x[slices] for slices in self.slices_list]

    def cat(self, x_list: List[Tensor]) -> Tensor:
        """Reverses the `split` method, suitable for the reverse process."""
        item = x_list[-1]
        shape = (item.shape[0], *self.metadata_list[-1][0])
        x = torch.zeros(shape, dtype=item.dtype, device=item.device)
        for item, slices in zip(x_list, self.slices_list):
            x[slices] = item
        return x

    def stepwise_cat(
        self, x_frozen: Tensor, x_active: Tensor, ind: int
    ) -> Tensor:
        """Concatenates `x_frozen` and `x_active` in order to create
        an updated frozen variable for the next layer, suitable for the forward
        process.

        The inputs are defined as::

            x_active[n] = split(var)[n]
            x_frozen[n] = stepwise_cat(x_frozen[n - 1], x_active[n - 1])

        with `x_frozen[0] = None`.
        """
        if ind == 0:
            return x_active

        lat_shape, axis = self.metadata_list[ind][:2]
        pre_axes = [slice(None)] * (1 + axis)  # 1 for the batch axis
        shape = (x_active.shape[0], *lat_shape)

        x = torch.zeros(shape, dtype=x_active.dtype, device=x_active.device)
        x[pre_axes + [slice(0, None, 2)]] = x_frozen
        x[pre_axes + [slice(1, None, 2)]] = x_active

        return x

    def stepwise_split(self, x: Tensor, ind: int) -> [Tensor, Tensor]:
        """Reverses the `stepwise_cat` method, i.e., splits the input to active
        and frozen parts, suitable for the reverse process.
        """

        if ind == 0:
            return None, x

        _, axis = self.metadata_list[ind][:2]
        pre_axes = [slice(None)] * (1 + axis)    # 1 for the batch axis

        x_frozen = x[pre_axes + [slice(0, None, 2)]]
        x_active = x[pre_axes + [slice(1, None, 2)]]

        return x_frozen, x_active

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
def test_fibo_splitter(shape=(4, 13)):
    """Create and test FibonacciTilingSplitter."""
    splitter = FibonacciTilingSplitter(shape)
    print(splitter)

    x = torch.randn(2, *shape)

    print("\nChecking if splitter.cat is inverse of splitter.split:")
    x_list = splitter.split(x)
    print(splitter.cat(x_list) / x)

    print("\nChecking if stepwise_split is inverse of stepwise_cat:")
    a, b = x_list[:2]
    y = splitter.stepwise_cat(a, b, 1)
    c, d = splitter.stepwise_split(y, 1)
    print((c / a).ravel())
    print((d / b).ravel())
