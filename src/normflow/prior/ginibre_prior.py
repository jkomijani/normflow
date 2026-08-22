# Copyright (c) 2021-2026 Javad Komijani

"""This module is for introducing priors for random matrices."""


from .prior import Prior
from ..lib.stats import GinibreCMatrixDist


class GinibrePrior(Prior):
    """Generate random matrices from the Ginibre ensemble.

    Parameters
    ----------
    n : int
        Dimension of the Ginibre matrices.
    shape : Tuple[int] | None (optional, default=None)
        Lattice shape, i.e. the shape of tensor of matrices sampled
        per configuration; treated as () if None. The batch axis is
        added separately by `batch_size` in `sample`/`sample_`.
    sigma : float (optional, default=1)
        Standard deviation of the underlying normal distribution.
    """

    def __init__(self, *, n, shape=None, sigma=1, **kwargs):
        if shape is None:
            shape = ()
        dist = GinibreCMatrixDist(n=n, shape=shape, sigma=sigma)
        super().__init__(dist, **kwargs)
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
