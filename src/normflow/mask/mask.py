# Copyright (c) 2021-2023 Javad Komijani

"""This module includes utilities for masking inputs.

Each mask must have three methods:
    1. split (to partition data to two parts),
    2. cat (to put the partitions together),
    3. purify (to make sure there is no contamination from other partition).
"""

import torch
import itertools

from abc import abstractmethod, ABC


class Mask(torch.nn.Module, ABC):
    """Applies the given mask of 0s and 1s."""

    def __init__(self, **mask_kwargs):
        super().__init__()
        mask = self.make_mask(**mask_kwargs)
        self.register_buffer('_mask', mask)
        self.register_buffer('_c_mask', 1 - mask)
        self.mask_kwargs = mask_kwargs

    def __str__(self):
        return self._mask.__str__()

    def split(self, x):
        return self._mask * x, self._c_mask * x

    def cat(self, x_0, x_1):
        return x_0 + x_1

    def purify(self, x_chnl, channel):
        return x_chnl * (self._mask if channel == 0 else self._c_mask)

    @staticmethod
    @abstractmethod
    def make_mask():
        pass


class EvenOddMask(Mask):
    """Creates an even-odd mask of given shape and parity.

    One can exclude a specific direction by providing a value to `exclude_mu`,
    which is by default None. Then the mask in direction of `exclude_mu` is
    constant.
    """

    @staticmethod
    def make_mask(*, shape, parity=0, exclude_mu=None):
        mask = torch.empty(shape, dtype=torch.uint8)
        for ind in itertools.product(*tuple([range(l) for l in shape])):
            if exclude_mu is None:
                mask[ind] = (1 - parity + sum(ind)) % 2
            else:
                mask[ind] = (1 - parity + sum(ind) - ind[exclude_mu]) % 2
        return mask


class FourWayParityMask(Mask):
    """
    Partition a tensor into 4 disjoint boolean masks via:
    (1) an even/odd split along one axis, and
    (2) a checkerboard pattern over all indices.

    `mask_id ∈ {0,1,2,3}` selects the combination of stripe (axis split)
    and checkerboard phase.
    """

    @staticmethod
    def make_mask(*, shape, mask_id: int, zebra_mu: int):
        """
        Args:
            shape:
            mask_id (int): encodes two bits:
                - parity: which checkerboard phase to use
                - visible_ind: which half (even/odd) along zebra_mu to keep
            zebra_mu (int): Axis along which to apply the zebra split.
        """
        parity = mask_id % 2
        visible_ind = mask_id // 2

        # uint8 for compatibility with existing code; bool would be preferable
        mask = torch.zeros(shape, dtype=torch.uint8)

        # iterate over all indices (slow but explicit reference implementation)
        for ind in itertools.product(*[range(l) for l in shape]):
            # keep only even/odd slice along zebra_mu
            if ind[zebra_mu] % 2 == visible_ind:
                # checkerboard pattern: sum(ind) % 2, shifted by parity
                mask[ind] = (1 - parity + sum(ind)) % 2

        return mask


class AlongAxesEvenOddMask(Mask):
    """Creates a mask that alternates only in a specific given direction."""

    @staticmethod
    def make_mask(*, shape, parity=0, mu=0):
        mask = torch.empty(shape, dtype=torch.uint8)
        for ind in itertools.product(*tuple([range(l) for l in shape])):
            mask[ind] = (1 - parity + ind[mu]) % 2
        return mask


class DummyMask:

    def __init__(self, parity=0):
        self.parity = parity

    def split(self, x):
        if self.parity == 0:
            return x, None
        else:
            return None, x

    def cat(self, x_0, x_1):
        if self.parity == 0:
            return x_0
        else:
            return x_1

    @staticmethod
    def purify(x_chnl, *args, **kwargs):
        return x_chnl
