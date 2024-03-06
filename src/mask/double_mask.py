# Copyright (c) 2021-2023 Javad Komijani

"""This module includes utilities for masking inputs.

Each mask must have three methods:
    1. split (to partition data to two parts),
    2. cat (to put the partitions together),
    3. purify (to make sure there is no contamination from other partition).
"""

import torch

from .mask import EvenOddMask
from .partitioner import AlongAxisEvenOddPartitioner


class DoubleMask(torch.nn.Module):
    """Consistes of two masks: the first one makes a part of the data
    invisible, and the second one acts only on the visible data.

    The invisible data will be saved and will be used once the `cat` method is
    called.
    """

    def __init__(self, *, invisibility_mask, outer_mask):
        super().__init__()
        self.invisibility_mask = invisibility_mask
        self.outer_mask = outer_mask

    def split(self, x):
        x, self._x_invisible = self.invisibility_mask.split(x)
        return self.outer_mask.split(x)

    def cat(self, x_0, x_1):
        x = self.outer_mask.cat(x_0, x_1)
        return self.invisibility_mask.cat(x, self._x_invisible)

    def purify(self, x_chnl, channel, **kwargs):
        outer_purified = self.outer_mask.purify(x_chnl, channel, **kwargs)
        return self.invisibility_mask.purify(outer_purified, channel=0)


class GaugeLinksDoubleMask(DoubleMask):

    def __init__(self, *, shape, parity, mu):
        mask0 = EvenOddMask(shape=shape, parity=parity, exclude_mu=mu)
        mask1 = AlongAxisEvenOddPartitioner(mu)
        super().__init__(invisibility_mask=mask0, outer_mask=mask1)
        self._mask0_splitted = mask0._mask[mask1.even_ind[1:]]

    def purify(self, x_chnl, channel, **kwargs):
        outer_purified = self.outer_mask.purify(x_chnl, channel, **kwargs)
        return outer_purified * self._mask0_splitted
