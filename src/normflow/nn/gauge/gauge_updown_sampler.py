# Copyright (c) 2026 Javad Komijani & Lara Turgut

"""
Gauge-equivariant up and down sampling of lattice gauge links.
"""

from typing import List, Tuple
import torch


__all__ = ["GaugeEquivDownsampler"]


# =============================================================================
class GaugeEquivDownsampler(torch.nn.Module):
    """
    Gauge-equivarinat, invertible downsampling module with skip connection.

    The forward pass returns a downsampled tensor along with the original
    input as a skip connection. The reverse pass reconstructs the input
    using the upsampler and the stored skip tensor.
    """

    def forward(self, x):
        """Downsample input and return (x_down, skip=input)."""
        x_down = gauge_equivariant_downsampler(x)
        return x_down, x

    def reverse(self, x_down, x):
        """Reconstruct input from (x_down, skip) using the upsampler."""
        return gauge_equivariant_upsampler(x_down, x)


# =============================================================================
def gauge_equivariant_downsampler(
    x_fine: torch.Tensor,
    dims: Tuple[int] | None = None,
    prefix_dims: int = 1,
    sites_before_link: bool = True
) -> torch.Tensor:
    """
    Gauge-equivariant downsampling of lattice gauge links.

    This operation performs a blocking transformation that reduces the lattice
    resolution by a factor of two along selected spatial dimensions. Coarse
    gauge links are constructed by multiplying pairs of adjacent fine links
    along each direction μ.

    The coarse μ-link is defined as

        U_coarse,μ(x_even) = U_fine,μ(x_even) @ U_fine,μ(x_even + μ)

    where x_even denotes a lattice site whose coordinates are even along all
    blocked dimensions.

    Parameters
    ----------
    x_fine : torch.Tensor
        Tensor containing the gauge links. After any batch and channel axes,
        the spatial lattice axes come first (if sites_before_link=True),
        followed by the link direction axis, and then the matrix components.
    dims : tuple[int] or None, optional
        Axes to downsample, corresponding to spatial dimensions in the lattice.
        If None (default), all spatial dimensions are downsampled.
    prefix_dims : int, default=1
        Number of leading batch and channel dimensions in the tensor.
        For example, if x.shape = (batch, channel, Lx, Ly, Lz, Lt, mu, Nc, Nc),
        then prefix_dims=2. If only a single batch dimension, prefix_dims=1.
    sites_before_link : bool, default=True
        Whether the spatial lattice axes come before the link axis.

    Returns
    -------
    torch.Tensor
        Coarse lattice gauge links with lattice extents halved along the
        selected dimensions.
    """
    # Determine link axis
    link_axis = -3 if sites_before_link else prefix_dims
    links = torch.unbind(x_fine, dim=link_axis)

    # Determine number of spatial dimensions (exclude batch/direction/matrix)
    spatial_ndim = x_fine.ndim - prefix_dims - 3

    # Initialize a list to store downsamples for each direction 'mu'
    coarse_stack: List[torch.Tensor] = [None] * spatial_ndim

    downsampling_dims = range(spatial_ndim) if dims is None else dims
    for d in downsampling_dims:
        if d < 0 or d >= spatial_ndim:
            raise ValueError("Invalid spatial dimension in dims.")

    # Loop over each link direction 'mu'
    for mu in range(spatial_ndim):
        link_mu = links[mu]
        idx0 = [slice(None)] * link_mu.ndim
        idx1 = [slice(None)] * link_mu.ndim
        for nu in downsampling_dims:
            dim = prefix_dims + nu
            if link_mu.shape[dim] % 2 != 0:
                raise ValueError("All lattice dimensions must be even.")
            idx0[dim] = slice(0, None, 2)
            idx1[dim] = slice(1 if mu == nu else 0, None, 2)
        idx0 = tuple(idx0)
        idx1 = tuple(idx1)
        coarse_stack[mu] = link_mu[idx0] @ link_mu[idx1]

    # Stack coarse directions back into a link axis at the proper place
    return torch.stack(coarse_stack, dim=link_axis)


# =============================================================================
def gauge_equivariant_upsampler(
    x_coarse: torch.Tensor,
    x_fine: torch.Tensor,
    dims: Tuple[int] | None = None,
    prefix_dims: int = 1,
    sites_before_link: bool = True,
    invertible: bool = True
) -> torch.Tensor:
    """
    Gauge-equivariant upsampling of lattice gauge links.

    This operation reconstructs fine links from an updated coarse lattice,
    using the original fine links provided through a skip connection
    (as in a U-Net architecture).

    Let a and b denote the two fine links that were originally blocked into
    a coarse link

        A = a @ b

    After a transformation produces a new coarse link A', the fine links are
    reconstructed as

        a' = A' @ b†
        b' = b  if inverible else (a† @ A')

    which ensures that the reconstruction transforms correctly under gauge
    transformations.

    Parameters
    ----------
    x_coarse : torch.Tensor
        Coarse lattice gauge links after transformation on the coarse lattice.
    x_fine : torch.Tensor
        Fine lattice gauge links from the encoder path (skip connection),
        representing the fine links *before* the coarse update. These are used
        to reconstruct the updated fine links.
    dims : tuple[int] or None, optional
        Spatial axes along which the lattice was previously downsampled.
        If None, all spatial dimensions are assumed to have been blocked.
    prefix_dims : int, default=1
        Number of leading batch and channel dimensions in the tensor.
        For example, if x.shape = (batch, channel, Lx, Ly, Lz, Lt, mu, Nc, Nc),
        then prefix_dims=2. If only a single batch dimension, prefix_dims=1.
    sites_before_link : bool, default=True
        Whether the spatial lattice axes come before the link axis.
    invertible : bool, default=True
        Whether the transformation is invertible.

    Returns
    -------
    torch.Tensor
        Reconstructed fine lattice gauge links.
    """
    # Determine link axis
    link_axis = -3 if sites_before_link else prefix_dims
    fine_links = torch.unbind(x_fine, dim=link_axis)
    coarse_links = torch.unbind(x_coarse, dim=link_axis)

    # Determine number of spatial dimensions (exclude batch/direction/matrix)
    spatial_ndim = x_fine.ndim - prefix_dims - 3

    # Initialize a list to store downsamples for each direction 'mu'
    fine_stack: List[torch.Tensor] = [None] * spatial_ndim

    upsampling_dims = range(spatial_ndim) if dims is None else dims
    for d in upsampling_dims:
        if d < 0 or d >= spatial_ndim:
            raise ValueError("Invalid spatial dimension in dims.")

    # Loop over each link direction 'mu'
    for mu in range(spatial_ndim):
        x = fine_links[mu].clone()
        y = coarse_links[mu]

        idx0 = [slice(None)] * x.ndim
        idx1 = [slice(None)] * x.ndim
        for nu in upsampling_dims:
            if x.shape[prefix_dims + nu] != 2 * y.shape[prefix_dims + nu]:
                raise ValueError("Coarse lattice must be half the fine one.")
            dim = prefix_dims + nu
            idx0[dim] = slice(0, None, 2)
            idx1[dim] = slice(1 if mu == nu else 0, None, 2)
        idx0 = tuple(idx0)
        idx1 = tuple(idx1)
        a_prime = y @ x[idx1].adjoint()
        x[idx0] = a_prime
        if not invertible:
            b_prime = x[idx0].adjoint() @ y
            x[idx1] = b_prime
        fine_stack[mu] = x

    # Stack coarse directions back into a link axis at the proper place
    return torch.stack(fine_stack, dim=link_axis)


# =============================================================================
def _test_gauge_equivaraince():
    """Shows the gauge equivariance of the down and up samplers."""

    # pylint: disable=import-outside-toplevel
    from normflow.prior import UniformSUnPrior

    Downsampler = GaugeEquivDownsampler()

    shape = (6, 6, 6, 6, 4)  # 2^4 lattice; the last axis is the "mu" axis.
    prior = UniformSUnPrior(3, shape=shape)

    # Define `x` and transform it with down and up samplers
    x = prior.sample(5)
    y, skip = Downsampler(x)
    z = Downsampler.reverse(y, skip)

    print(x.shape, y.shape, z.shape, (z - x).abs().mean())

    # Now gauge transform `x`; only the links connected to the origin
    q = prior.sample(1)[0, 0, 0, 0, 0, 0]
    for i in range(4):
        x[0, 0, 0, 0, 0, i] = q @ x[0, 0, 0, 0, 0, i]
    x[0, -1, 0, 0, 0, 0] = x[0, -1, 0, 0, 0, 0] @ q.adjoint()
    x[0, 0, -1, 0, 0, 1] = x[0, 0, -1, 0, 0, 1] @ q.adjoint()
    x[0, 0, 0, -1, 0, 2] = x[0, 0, 0, -1, 0, 2] @ q.adjoint()
    x[0, 0, 0, 0, -1, 3] = x[0, 0, 0, 0, -1, 3] @ q.adjoint()

    # Use the gauge transformed x & transform it with down sampler
    z, skip = Downsampler(x)

    # Undo the gauge transformation on `z` to check the gauge equivarience.
    for i in range(4):
        z[0, 0, 0, 0, 0, i] = q.adjoint() @ z[0, 0, 0, 0, 0, i]
    z[0, -1, 0, 0, 0, 0] = z[0, -1, 0, 0, 0, 0] @ q
    z[0, 0, -1, 0, 0, 1] = z[0, 0, -1, 0, 0, 1] @ q
    z[0, 0, 0, -1, 0, 2] = z[0, 0, 0, -1, 0, 2] @ q
    z[0, 0, 0, 0, -1, 3] = z[0, 0, 0, 0, -1, 3] @ q

    print(f"Gauge Equivariant if {(z - y).abs().mean()} is approximately 0")
