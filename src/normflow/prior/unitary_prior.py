# Copyright (c) 2021-2026 Javad Komijani

"""This module is for introducing unitary priors."""

from typing import Tuple

from .prior import Prior
from ..lib.stats import UnGroup, SUnGroup, DiagonalSUnGroup, U1Group


__all__ = [
    "UniformUnPrior",
    "UniformSUnPrior",
    "UniformDiagonalSUnPrior",
    "UniformU1Prior",
    "UnPrior", "SUnPrior", "U1Prior",  # alias for legacy
]


class UniformUnPrior(Prior):
    """Generate unitary matrices uniformly with the Haar measure.

    Parameters
    ----------
    n : int
        Dimension of the U(n) matrices.
    shape : Tuple[int] | None (optional, default=None)
        Lattice shape, i.e. the shape of tensor of matrices sampled
        per configuration; treated as () if None. The batch axis is
        added separately by `batch_size` in `sample`/`sample_`.
    drop_constant_log_prob : bool (optional, default=False)
        If True, log_prob returns zeros instead of -log(volume).
    """

    def __init__(
        self,
        n: int,
        shape: Tuple | None = None,
        drop_constant_log_prob: bool = False,
        **super_kwargs
    ):
        if shape is None:
            shape = ()

        dist = UnGroup(
            n, shape=shape, drop_constant_log_prob=drop_constant_log_prob
        )

        super().__init__(dist, **super_kwargs)

        self.shape = shape

    def to(self, *args, **kwargs):
        """Move the distribution parameters to a device, implying that
        the samples will also be created on the same device.
        """
        dist = self.dist.normal_dist
        dist.loc = dist.loc.to(*args, **kwargs)
        dist.scale = dist.scale.to(*args, **kwargs)

    @property
    def parameters(self):
        """Returns all parameters needed to define the prior in a dict."""
        dist = self.dist.normal_dist
        return {'loc': dist.loc, 'scale': dist.scale}


class UniformSUnPrior(Prior):
    """Generate SU(n) matrices uniformly with the Haar measure.

    Parameters
    ----------
    n : int
        Dimension of the SU(n) matrices.
    shape : Tuple[int] | None (optional, default=None)
        Lattice shape, i.e. the shape of tensor of matrices sampled
        per configuration; treated as () if None. The batch axis is
        added separately by `batch_size` in `sample`/`sample_`.
    drop_constant_log_prob : bool (optional, default=False)
        If True, log_prob returns zeros instead of -log(volume).
    """

    def __init__(
        self,
        n: int,
        shape: Tuple | None = None,
        drop_constant_log_prob: bool = False,
        **super_kwargs
    ):
        if shape is None:
            shape = ()

        dist = SUnGroup(
            n, shape=shape, drop_constant_log_prob=drop_constant_log_prob
        )

        super().__init__(dist, **super_kwargs)

        self.shape = shape

    def to(self, *args, **kwargs):
        """Move the distribution parameters to a device, implying that
        the samples will also be created on the same device.
        """
        dist = self.dist.normal_dist
        dist.loc = dist.loc.to(*args, **kwargs)
        dist.scale = dist.scale.to(*args, **kwargs)

    @property
    def parameters(self):
        """Returns all parameters needed to define the prior in a dict."""
        dist = self.dist.normal_dist
        return {'loc': dist.loc, 'scale': dist.scale}


class UniformDiagonalSUnPrior(Prior):
    """Generate diagonal SU(n) matrices with i.i.d. uniform phases.

    Each matrix lies in the maximal torus of SU(n): `n - 1` phases are
    drawn i.i.d. uniform on the circle, and the last is fixed so
    `det = 1`. This is *not* the eigenvalue distribution of a
    Haar-random SU(n) matrix -- see `rand_diagonal_sun_group_like`.

    Parameters
    ----------
    n : int
        Dimension of the SU(n) matrices.
    shape : Tuple[int] | None (optional, default=None)
        Lattice shape, i.e. the shape of tensor of matrices sampled
        per configuration; treated as () if None. The batch axis is
        added separately by `batch_size` in `sample`/`sample_`.
    drop_constant_log_prob : bool (optional, default=False)
        If True, log_prob returns zeros instead of -log(volume).
    """

    def __init__(
        self,
        n: int,
        shape: Tuple | None = None,
        drop_constant_log_prob: bool = False,
        **super_kwargs
    ):
        if shape is None:
            shape = ()

        dist = DiagonalSUnGroup(
            n, shape=shape, drop_constant_log_prob=drop_constant_log_prob
        )

        super().__init__(dist, **super_kwargs)

        self.shape = shape

    def to(self, *args, **kwargs):
        """Move the distribution parameters to a device, implying that
        the samples will also be created on the same device.
        """
        dist = self.dist.uniform_dist
        dist.low = dist.low.to(*args, **kwargs)
        dist.high = dist.high.to(*args, **kwargs)

    @property
    def parameters(self):
        """Returns all parameters needed to define the prior in a dict."""
        dist = self.dist.uniform_dist
        return {'low': dist.low, 'high': dist.high}


class UniformU1Prior(Prior):
    """Generate U(1) variables uniformly with the Haar measure.

    This is a faster implementation of random U(1) than `UniformUnPrior(n=1)`.

    Parameters
    ----------
    shape : Tuple[int] | None (optional, default=None)
        Lattice shape, i.e. the shape of tensor of variables sampled
        per configuration; treated as () if None. The batch axis is
        added separately by `batch_size` in `sample`/`sample_`.
    """
    def __init__(self, shape=None, **kwargs):
        if shape is None:
            shape = ()
        dist = U1Group(shape=shape)
        super().__init__(dist, **kwargs)
        self.shape = shape

    def to(self, *args, **kwargs):
        """Move the distribution parameters to a device, implying that
        the samples will also be created on the same device.
        """
        dist = self.dist.uniform_dist
        dist.low = dist.low.to(*args, **kwargs)
        dist.high = dist.high.to(*args, **kwargs)

    @property
    def parameters(self):
        """Returns all parameters needed to define the prior in a dict."""
        dist = self.dist.uniform_dist
        return {'low': dist.low, 'high': dist.high}


# aliased for legacy
UnPrior = UniformUnPrior
SUnPrior = UniformSUnPrior
U1Prior = UniformU1Prior
