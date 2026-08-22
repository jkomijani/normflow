# Copyright (c) 2021-2022 Javad Komijani


"""This module is for introducing priors..."""

import copy
from abc import abstractmethod, ABC

import torch
import numpy as np


class Prior(ABC):
    """A template class to initiate a prior distribution.

    Subclasses set `self.shape`, the lattice shape, i.e. the shape of
    the tensor sampled per configuration. The batch axis is separate:
    it is added by `batch_size` in `sample`/`sample_`, giving samples
    of shape `(batch_size, *shape)`.
    """

    propagate_density = False
    shape = ()

    def __init__(self, dist):
        self.dist = dist

    def sample(self, batch_size=1):
        """Return `batch_size` samples, each of shape `self.shape`."""
        return self.dist.sample((batch_size,))

    def sample_(self, batch_size=1):
        """Return samples and their log-probabilities; see `sample`."""
        x = self.dist.sample((batch_size,))
        return x, self.log_prob(x)

    def log_prob(self, x):
        """Return the log-probability of each sample in the batch."""
        log_prob_density = self.dist.log_prob(x)
        if self.propagate_density:
            return log_prob_density
        dim = range(1, len(log_prob_density.shape))  # 0: batch axis
        return torch.sum(log_prob_density, dim=tuple(dim))

    @property
    def nvar(self):
        """Number of variables in the lattice shape `self.shape`."""
        return np.prod(self.shape)

    @abstractmethod
    def to(self, *args, **kwargs):
        """
        Move the distribution parameters to a device, implying that the samples
        will also be created on the same device.
        """

    @property
    @abstractmethod
    def parameters(self):
        """Returns all parameters needed to define the prior in a dict."""


class UniformPrior(Prior):
    """Uniform prior with parameters `low` and `high`.

    Parameters
    ----------
    low, high : float | Tensor | None (optional, default=None)
        Bounds of the uniform distribution. If `shape` is given, they
        may be None (default 0 and 1 respectively), scalars, or
        broadcastable to `shape`. If `shape` is None, both must be
        given as tensors of the same shape, from which `shape` is
        inferred; if `low`, `high`, and `shape` are all None, `shape`
        is treated as ().
    shape : Tuple[int] | None (optional, default=None)
        Lattice shape, i.e. the shape of tensor of variables sampled
        per configuration. See `low`, `high` for how it interacts
        with them. The batch axis is added separately by `batch_size`
        in `sample`/`sample_`.
    """

    def __init__(self, low=None, high=None, shape=None, **kwargs):
        if shape is None and low is None and high is None:
            shape = ()

        # Default values if missing
        if low is None:
            low = 0
        if high is None:
            high = 1
        if shape is not None:
            # Broadcast to shape
            low = low + torch.zeros(shape)
            high = high * torch.ones(shape)
        else:
            shape = low.shape
        dist = torch.distributions.uniform.Uniform(low, high)
        super().__init__(dist, **kwargs)
        self.shape = shape

    def to(self, *args, **kwargs):
        self.dist.low = self.dist.low.to(*args, **kwargs)
        self.dist.high = self.dist.high.to(*args, **kwargs)

    @property
    def parameters(self):
        return {'low': self.dist.low, 'high': self.dist.high}


class NormalPrior(Prior):
    """Normal prior with parameters `loc` and `scale`.

    Parameters
    ----------
    loc, scale : float | Tensor | None (optional, default=None)
        Mean and standard deviation of the normal distribution. If
        `shape` is given, they may be None (default 0 and 1
        respectively), scalars, or broadcastable to `shape`. If
        `shape` is None, both must be given as tensors of the same
        shape, from which `shape` is inferred; if `loc`, `scale`, and
        `shape` are all None, `shape` is treated as ().
    shape : Tuple[int] | None (optional, default=None)
        Lattice shape, i.e. the shape of tensor of variables sampled
        per configuration. See `loc`, `scale` for how it interacts
        with them. The batch axis is added separately by `batch_size`
        in `sample`/`sample_`.
    """

    blockupdater = None

    def __init__(self, loc=None, scale=None, shape=None, **kwargs):
        if shape is None and loc is None and scale is None:
            shape = ()

        # Default values if missing
        if loc is None:
            loc = 0
        if scale is None:
            scale = 1
        if shape is not None:
            # Broadcast to shape
            loc = loc + torch.zeros(shape)
            scale = scale * torch.ones(shape)
        else:
            # Must already match in shape
            shape = loc.shape

        dist = torch.distributions.normal.Normal(loc, scale)
        super().__init__(dist, **kwargs)
        self.shape = shape

    def setup_blockupdater(self, block_len):
        """Set up an instance of BlockUpdater."""
        # For simplicity we assume that loc & scale are identical everywhere.
        chopped_prior = NormalPrior(
            loc=self.dist.loc.ravel()[:block_len],
            scale=self.dist.scale.ravel()[:block_len]
        )
        self.blockupdater = BlockUpdater(chopped_prior, block_len)

    def to(self, *args, **kwargs):
        self.dist.loc = self.dist.loc.to(*args, **kwargs)
        self.dist.scale = self.dist.scale.to(*args, **kwargs)

    @property
    def parameters(self):
        return {'loc': self.dist.loc, 'scale': self.dist.scale}


class PriorList:
    """Aggregate several priors and sample/evaluate them as a group."""

    def __init__(self, prior_list):
        self.prior_list = prior_list

    def sample(self, batch_size=1):
        """Sample from each prior in the list."""
        return [prior.sample(batch_size) for prior in self.prior_list]

    def sample_(self, batch_size=1):
        """Sample and return log-probabilities for each prior."""
        x = [prior.sample(batch_size) for prior in self.prior_list]
        return x, self.log_prob(x)

    def log_prob(self, x):
        """Return log-probability of x under each prior."""
        return [prior.log_prob(x_) for prior, x_ in zip(self.prior_list, x)]

    @property
    def nvar(self):
        """Total number of variables across all priors."""
        return sum(prior.nvar for prior in self.prior_list)

    def to(self, *args, **kwargs):
        """
        Move the distribution parameters to a device, implying that the samples
        will also be created on the same device.
        """
        for prior in self.prior_list:
            prior.to(*args, **kwargs)

    @property
    def parameters(self):
        """Returns all parameters needed to define the priors in a dict."""
        return [prior.parameters for prior in self.prior_list]


class BlockUpdater:
    """
    In-place block-wise resampler used to update part of a tensor with fresh
    draws from `chopped_prior`, `block_len` entries at a time.
    """

    def __init__(self, chopped_prior, block_len):
        self.block_len = block_len
        self.chopped_prior = chopped_prior
        self.backup_block = None

    def __call__(self, x, block_ind):
        """In-place updater"""
        batch_size = x.shape[0]
        view = x.view(batch_size, -1, self.block_len)
        self.backup_block = copy.deepcopy(view[:, block_ind])
        view[:, block_ind] = self.chopped_prior.sample(batch_size)

    def restore(self, x, block_ind, restore_ind=slice(None)):
        """Restore the previously overwritten block."""
        batch_size = x.shape[0]
        view = x.view(batch_size, -1, self.block_len)
        view[restore_ind, block_ind] = self.backup_block[restore_ind]
