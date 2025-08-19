# Copyright (c) 2021-2024 Javad Komijani

"""This module includes utilities for masking inputs.

Each mask must have three methods:
    1. split (to partition data to two parts),
    2. cat (to put the partitions together),
    3. purify (to make sure there is no contamination from other partition).
"""

import torch
import itertools

from .mask import Mask


class MatrixMask(Mask, ABC):
    """Applies the given mask of 0s and 1s to all matrices and then replaces
    the vanishing matrices with the identity matrix.
    """

    def split(self, x):
        eye = torch.eye(x.shape[-1], device=x.device)
        return (self._mask * x + self._c_mask * eye,
                self._c_mask * x + self._mask * eye)

    def cat(self, x_0, x_1):
        eye = torch.eye(x.shape[-1], device=x.device)
        return x_0 + x_1 - eye

    def purify(self, x_chnl, channel):
        eye = torch.eye(x.shape[-1], device=x.device)
        if channel == 0:
            return self._mask * x + self._c_mask * eye
        else:
            return self._c_mask * x + self._mask * eye


class EvenOddMatrixMask(MatrixMask):
    """Creates an even-odd matrix mask of given shape and parity.

    One can exclude a specific direction by providing a value to `exclude_mu`,
    which is by default None. Then the mask in direction of `exclude_mu` is
    constant.
    """

    @staticmethod
    def make_mask(*, shape, parity=0, exclude_mu=None):
        shape = [*shape, 1, 1]  # last 2 axes are for matrix space
        mask = torch.empty(shape, dtype=torch.uint8)
        for ind in itertools.product(*tuple([range(l) for l in shape])):
            if exclude_mu is None:
                mask[ind] = (1 - parity + sum(ind)) % 2
            else:
                mask[ind] = (1 - parity + sum(ind) - ind[exclude_mu]) % 2
        return mask


class AlongAxesEvenOddMatrixMask(MatrixMask):
    """Creates a mask that alternates only in a specific given direction."""

    @staticmethod
    def make_mask(*, shape, parity=0, mu=0):
        shape = [*shape, 1, 1]  # last 2 axes are for matrix space
        mask = torch.empty(shape, dtype=torch.uint8)
        for ind in itertools.product(*tuple([range(l) for l in shape])):
            mask[ind] = (1 - parity + ind[mu]) % 2
        return mask
