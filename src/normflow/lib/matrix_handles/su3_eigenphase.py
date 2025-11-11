# Copyright (c) 2022-2025 Javad Komijani

"""
SU(3) phase grid utilities.

Provides a class to generate SU(3) phase mesh grids and compute joint
and marginal probability distributions given a user-defined action.
"""

# pylint: disable=invalid-name

import math
import torch


__all__ = ["SU3PhaseGrid", "su3_phase_marginal_dist"]



class SU3PhaseGrid:
    """Generate SU(3) phase grids and compute joint probability densities.

    This class provides utilities to create a 2D mesh grid of SU(3) phases
    and to evaluate the joint PDF over the phase space given a user-defined
    action function.
    """

    @staticmethod
    def generate_grid(
        bins: int,
        xlim=(-math.pi*0.9999, math.pi*0.9999),
        ylim=(-math.pi, math.pi)
    ):
        """Create a mesh grid of SU(3) phases.

        Args:
            bins: Number of points along each axis.
            xlim: Limits for the X-axis phases.
            ylim: Limits for the Y-axis phases.

        Returns:
            X, Y, Z: Tensors representing the mesh grid of SU(3) phases.
        """
        x = torch.linspace(*xlim, bins)
        y = torch.linspace(*ylim, bins)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        Z = -(X + Y)
        return X, Y, Z

    @staticmethod
    def compute_joint_pdf(X, Y, Z, action):
        """Compute the joint probability density over the SU(3) phase grid.

        Args:
            X, Y, Z: Mesh grids of SU(3) phases.
            action: Function that computes the action for SU(3) matrices.

        Returns:
            Tensor of joint PDF values over the phase grid.
        """
        phase = torch.stack([X, Y, Z], dim=-1)
        eigvals = torch.exp(1j * phase).reshape(-1, 3)

        conj_vol = calc_conjugacy_vol(eigvals).reshape(*X.shape)
        action_val = action(torch.diag_embed(eigvals)).reshape(*X.shape)

        return conj_vol * torch.exp(-action_val)


def su3_phase_marginal_dist(action, bins: int = 200):
    """Compute the marginal PDF along the first SU(3) phase.

    The function generates a phase mesh grid, evaluates the joint PDF, and
    sums over the second axis to obtain a marginal distribution. The result
    is normalized to integrate to one over the grid.

    Args:
        action: Function computing the action for SU(3) matrices.
        bins: Number of bins along each axis for the phase grid.

    Returns:
        X[0]: Phase values along the first axis.
        marg_pdf: Normalized marginal PDF along the first phase.
    """
    delta_vol = (2 * math.pi / bins) ** 2

    X, Y, Z = SU3PhaseGrid.generate_grid(bins)
    joint_pdf = SU3PhaseGrid.compute_joint_pdf(X, Y, Z, action=action)

    marg_pdf = torch.sum(joint_pdf, dim=1)
    marg_pdf /= (torch.sum(marg_pdf) * delta_vol ** 0.5)

    return X[0], marg_pdf


def calc_conjugacy_vol(eigvals):
    """Return conjugacy volume up to a multiplacative constant."""
    def prod_abs_sq(x):
        return torch.prod(torch.abs(x)**2, dim=-1)
    vol = torch.ones(eigvals.shape[:-1], device=eigvals.device)
    for k in range(eigvals.shape[-1] - 1):
        vol *= prod_abs_sq(eigvals[..., k:k+1] - eigvals[..., k+1:])
    return vol.unsqueeze(-1)  # unsqueeze to keep dimensions the same
