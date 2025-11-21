# Copyright (c) 2021-2022 Javad Komijani


"""This module is for introducing priors..."""


import torch
import copy
import numpy as np

from abc import abstractmethod, ABC


class Prior(ABC):
    """A template class to initiate a prior distribution."""

    propagate_density = False

    def __init__(self, dist, seed=None):
        self.dist = dist
        Prior.manual_seed(seed)

    def sample(self, batch_size=1):
        return self.dist.sample((batch_size,))

    def sample_(self, batch_size=1):
        x = self.dist.sample((batch_size,))
        return x, self.log_prob(x)

    def log_prob(self, x):
        log_prob_density = self.dist.log_prob(x)
        if self.propagate_density:
            return log_prob_density
        else:
            dim = range(1, len(log_prob_density.shape))  # 0: batch axis
            return torch.sum(log_prob_density, dim=tuple(dim))

    @staticmethod
    def manual_seed(seed):
        if isinstance(seed, int):
            torch.manual_seed(seed)

    @property
    def nvar(self):
        return np.prod(self.shape)

    @abstractmethod
    def to(self, *args, **kwargs):
        """
        Moves the distibution parameters to a device, implying that the samples
        will also be created on the same device.
        """
        pass

    @property
    @abstractmethod
    def parameters(self):
        """Returns all parameters needed to define the prior in a dict."""
        pass


class UniformPrior(Prior):
    """Uniform prior with parameters `low` and `high`.

    If `shape` is provided, `low` and `high` may be None, scalars, or
    broadcastable to `shape`. If `shape` is None, both must be provided
    and have the same shape.
    """

    def __init__(self, low=None, high=None, shape=None, seed=None, **kwargs):
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
        super().__init__(dist, seed, **kwargs)
        self.shape = shape

    def to(self, *args, **kwargs):
        """
        Moves the distibution parameters to a device, implying that the samples
        will also be created on the same device.
        """
        self.dist.low = self.dist.low.to(*args, **kwargs)
        self.dist.high = self.dist.high.to(*args, **kwargs)

    @property
    def parameters(self):
        """Returns all parameters needed to define the prior in a dict."""
        return dict(low=self.dist.low, high=self.dist.high)


class NormalPrior(Prior):
    """Normal prior with parameters `loc` and `scale`.

    If `shape` is provided, `loc` and `scale` may be None, scalars, or
    broadcastable to `shape`. If `shape` is None, both must be provided
    and have the same shape.
    """

    def __init__(self, loc=None, scale=None, shape=None, seed=None, **kwargs):
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
        super().__init__(dist, seed, **kwargs)
        self.shape = shape

    def setup_blockupdater(self, block_len):
        # For simplicity we assume that loc & scale are identical everywhere.
        chopped_prior = NormalPrior(
                loc=self.dist.loc.ravel()[:block_len],
                scale=self.dist.scale.ravel()[:block_len]
                )
        self.blockupdater = BlockUpdater(chopped_prior, block_len)

    def to(self, *args, **kwargs):
        """
        Moves the distibution parameters to a device, implying that the samples
        will also be created on the same device.
        """
        self.dist.loc = self.dist.loc.to(*args, **kwargs)
        self.dist.scale = self.dist.scale.to(*args, **kwargs)

    @property
    def parameters(self):
        """Returns all parameters needed to define the Prior in a dict."""
        return dict(loc=self.dist.loc, scale=self.dist.scale)


class PriorList:

    def __init__(self, prior_list):
        self.prior_list = prior_list

    def sample(self, batch_size=1):
        return [prior.sample(batch_size) for prior in self.prior_list]

    def sample_(self, batch_size=1):
        x = [prior.sample(batch_size) for prior in self.prior_list]
        return x, self.log_prob(x)

    def log_prob(self, x):
        return [prior.log_prob(x_) for prior, x_ in zip(self.prior_list, x)]

    @property
    def nvar(self):
        return sum([prior.nvar for prior in self.prior_list])

    def to(self, *args, **kwargs):
        """
        Moves the distibution parameters to a device, implying that the samples
        will also be created on the same device.
        """
        for prior in self.prior_list:
            prior.to(*args, **kwargs)

    @property
    def parameters(self):
        """Returns all parameters needed to define the priors in a dict."""
        return [prior.parameters for prior in self.prior_list]


class BlockUpdater:

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
        batch_size = x.shape[0]
        view = x.view(batch_size, -1, self.block_len)
        view[restore_ind, block_ind] = self.backup_block[restore_ind]
