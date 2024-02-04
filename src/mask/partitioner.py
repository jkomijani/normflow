# Copyright (c) 2023 Javad Komijani

"""Similar to mask, except for the shape of the input and output tensors."""

import torch


class ListPartitioner:

    @staticmethod
    def split(x):
        return x[0], x[1]

    @staticmethod
    def cat(x0, x1):
        return [x0, x1]

    @staticmethod
    def purify(x_chnl, *args, **kwargs):
        return x_chnl


class ChunkCatPartitioner:
    """For chunking the input along the chunk_axis, which must be positive.

    Parameters
    ----------
    chunk_axis : (positive) int
        specifies the axis along which the input is going to be partitioned.
    batch_ndim : int, optional
        specifies the number of batch axes in the data (default is 1).
    """

    def __init__(self, chunk_axis, batch_ndim=1):
        self.axis = chunk_axis + batch_ndim

    def split(self, x):
        return torch.chunk(x, 2, self.axis)

    def cat(self, x0, x1):
        return torch.cat([x0, x1], dim=self.axis)

    @staticmethod
    def purify(x_chnl, *args, **kwargs):
        return x_chnl


class AlongAxisEvenOddPartitioner:
    """For even odd slicing the input along the even_odd_axis, which must be
    positive.

    Parameters
    ----------
    even_odd_axis : (positive) int
        specifies the axis along which the input is going to be sliced.
    batch_ndim : int, optional
        specifies the number of batch axes in the data (default is 1).
    """

    def __init__(self, even_odd_axis, batch_ndim=1):
        self.axis = even_odd_axis + batch_ndim
        self.even_ind = [slice(None)] * self.axis + [slice(0, None, 2)]
        self.odd_ind = [slice(None)] * self.axis + [slice(1, None, 2)]

    def split(self, x):
        return x[self.even_ind], x[self.odd_ind]

    def cat(self, x_even, x_odd):
        shape = list(x_even.shape) 
        shape[self.axis] = x_even.shape[self.axis] + x_odd.shape[self.axis]
        x = torch.zeros(shape, dtype=x_even.dtype, device=x_even.device)
        x[self.even_ind] = x_even
        x[self.odd_ind] = x_odd
        return x

    @staticmethod
    def purify(x_chnl, *args, **kwargs):
        return x_chnl
