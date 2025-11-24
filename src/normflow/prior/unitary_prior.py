# Copyright (c) 2021-2025 Javad Komijani

"""This module is for introducing unitary priors."""

from typing import Tuple
import torch

from .prior import Prior, NormalPrior
from ..lib.stats import UnGroup, SUnGroup, U1Group
from ..lib.matrix_handles import SU2Algebra2Group_
from ..lib.matrix_handles import SU3Algebra2Group_


__all__ = [
    "UniformUnPrior", "UniformSUnPrior", "UniformU1Prior",
    "UnPrior", "SUnPrior", "U1Prior",  # alias for legacy
    "NormalSUnPrior"
]


class UniformUnPrior(Prior):
    """Generate unitary matrices uniformly with the Haar measure."""

    def __init__(
        self,
        n: int,
        shape: Tuple = (1,),
        drop_constant_log_prob: bool = False,
        **super_kwargs
    ):
        kws = dict(shape=shape, drop_constant_log_prob=drop_constant_log_prob)
        dist = UnGroup(n, **kws)

        super().__init__(dist, **super_kwargs)

        self.shape = shape

    def to(self, *args, **kwargs):
        """
        Moves the distibution parameters to a device, implying that the samples
        will also be created on the same device.
        """
        dist = self.dist.normal_dist
        dist.loc = dist.loc.to(*args, **kwargs)
        dist.scale = dist.scale.to(*args, **kwargs)

    @property
    def parameters(self):
        """Returns all parameters needed to define the prior in a dict."""
        dist = self.dist.normal_dist
        return dict(loc=dist.loc, scale=dist.scale)


class UniformSUnPrior(Prior):
    """Generate SU(n) matrices uniformly with the Haar measure."""

    def __init__(
        self,
        n: int,
        shape: Tuple = (1,),
        drop_constant_log_prob: bool = False,
        **super_kwargs
    ):
        kws = dict(shape=shape, drop_constant_log_prob=drop_constant_log_prob)
        dist = SUnGroup(n, **kws)

        super().__init__(dist, **super_kwargs)

        self.shape = shape

    def to(self, *args, **kwargs):
        """
        Moves the distibution parameters to a device, implying that the samples
        will also be created on the same device.
        """
        dist = self.dist.normal_dist
        dist.loc = dist.loc.to(*args, **kwargs)
        dist.scale = dist.scale.to(*args, **kwargs)

    @property
    def parameters(self):
        """Returns all parameters needed to define the prior in a dict."""
        dist = self.dist.normal_dist
        return dict(loc=dist.loc, scale=dist.scale)


class UniformU1Prior(Prior):
    """Generate U(1) variables uniformly with the Haar measure.

    This is a faster implementation of random U(1) than `UnPrior(n=1)`.
    """
    def __init__(self, shape=(1,), **kwargs):
        dist = U1Group(shape=shape)
        super().__init__(dist, **kwargs)
        self.shape = shape

    def to(self, *args, **kwargs):
        """
        Moves the distibution parameters to a device, implying that the samples
        will also be created on the same device.
        """
        dist = self.dist.uniform_dist
        dist.loc = dist.loc.to(*args, **kwargs)
        dist.scale = dist.scale.to(*args, **kwargs)

    @property
    def parameters(self):
        """Returns all parameters needed to define the prior in a dict."""
        dist = self.dist.uniform_dist
        return dict(loc=dist.loc, scale=dist.scale)


class NormalSUnPrior(NormalPrior):
    """Generate SU(n) matrices by exponentiating normal-distributed algebra
    elements.

    This class is a subclass of `NormalPrior` where the innermost dimension of
    the shape is fixed to `n^2 - 1``, the dimension of the Lie algebra `su(n)`.
    Samples drawn from the underlying normal distribution are interpreted as
    algebra elements and then mapped to SU(n) via `alg_to_grp`. The associated
    log-Jacobian correction is applied both when sampling and when evaluating
    log-probabilities.

    Args:
        n: Dimension of the SU(n) group (supports n=2 or n=3).
        shape: Batch shape for sampling algebra elements.
        super_kwargs: Passed to ``NormalPrior``. May include:
            * ``loc``: Mean of the underlying normal distribution.
            * ``scale``: Stddev of the underlying normal distribution.
            * ``seed``: Random seed for reproducible sampling.
    """
    def __init__(self, n: int, shape: Tuple = (1,), **super_kwargs):

        super().__init__(shape=(*shape, n**2 - 1), **super_kwargs)

        alg_to_grp_kwargs = {
            'coordinate_representation': True, 'makesure_invertible': False
        }
        if n == 2:
            self.alg_to_grp = SU2Algebra2Group_(**alg_to_grp_kwargs)
        elif n == 3:
            self.alg_to_grp = SU3Algebra2Group_(**alg_to_grp_kwargs)
        else:
            raise ValueError("Only n = 2, 3 are supported.")

    def sample(self, batch_size: int = 1):
        """Return samples of SU(n) matrices.

        Args:
            batch_size: Number of samples.
        """
        return self.sample_(batch_size)[0]

    def sample_(self, batch_size: int = 1):
        """Return samples of SU(n) matrices and log probabilites.

        Args:
            batch_size: Number of samples.
        """
        x, logr = super().sample_(batch_size)  # from normal distribution
        x, logj = self.alg_to_grp(x)
        return x, logr - logj

    # def log_prob(self, x):  do NOT do it: the super one needed in NormalPrior


# aliased for legace
UnPrior = UniformUnPrior
SUnPrior = UniformSUnPrior
U1Prior = UniformU1Prior
