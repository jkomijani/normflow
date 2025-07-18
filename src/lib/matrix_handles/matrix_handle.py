# Copyright (c) 2021-2024 Javad Komijani

"""This module has utilities to deal with eigenvalues of matrices.

For SU(n) matrices, the main class is `SUnMatrixParametrizer` that has several
methods, including methods to map matrices to phases (of eigenvalues) and
parametrize the phases. The reverse of these maps are also defined.
"""


import torch
import numpy as np

from .ordering import ZeroSumOrder, ModalOrder
from ..linalg import eigu  # eig for unitray matrices
from ..linalg import inverse_eign


pi = np.pi


# =============================================================================
class UnMatrixParametrizer:

    def __init__(self):
        self.eigvecs = None
        self.phase = None  # we save the phases/angles of the eigenvalues
        self.order = None  # an object to sort the eigen-phases/angles

    def free_memory(self):
        self.eigvecs = None
        self.phase = None
        self.order = None

    def matrix2phase_(self, matrix):
        """Preform the spectral decomposition of the input matrix, save the
        modal matrix, calculate the angles of eigenvalues, save and return
        them, Moreover, the logarithm of Jacobian of transformation will be
        returned.

        Here, `logJ` is the Jacobian of partitioning an integration over SU(n)
        matrices to integrals over corresponding spectral and modal matrices.
        The inverse of Jacobian is equal to the volume of conjugacy class.
        """
        eigvals, self.eigvecs = eigu(matrix)
        self.phase = torch.angle(eigvals)  # in (-pi, pi]
        # Note: when |eig| = 1, logJ of eig to phase conversion is zero;
        # thus, we only need to take care of the Jacobian of spectral
        # decomposition:
        # *inverse* of Jacobian equals the volume of conjugacy class
        logJ = -sum_density(self.calc_log_conjugacy_vol(eigvals))
        return self.phase, logJ

    def phase2matrix_(self, phase, reduce_=False):
        """Return the matrix corresponding to the input `phase` and logJ of
        transformation. (Inverse of `self.matrix2phase_`.)

        For the sake of frugal computing, the `reduce_` option is introduced
        such that if True, this method returns :math:`M  M_{old}^\dagger`,
        where :math:`M_{old}` is the matrix constructed with self.phases.
        """
        eigvals = torch.exp(1j * phase)
        eigvecs = self.eigvecs
        eig_red = eigvals * torch.exp(-1j * self.phase) if reduce_ else eigvals
        matrix = inverse_eign(eig_red, eigvecs)
        # Jacobian equals the volume of conjugacy class
        logJ = sum_density(self.calc_log_conjugacy_vol(eigvals))

        return matrix, logJ

    def phase2param_(self, *args, **kwargs):
        pass

    def param2phase_(self, *args, **kwargs):
        pass

    def matrix2param_(self, matrix):
        """Return a unique parametrization of the phase of eigenvalues, and
        logJ of the transformation.
        """
        phase, logJ_m2f = self.matrix2phase_(matrix)  # phase in (-pi, pi]
        param, logJ_f2p = self.phase2param_(phase)
        return param, logJ_m2f + logJ_f2p

    def param2matrix_(self, param, reduce_=False):
        """Return the matrix corresponding to `param` and logJ of
        transformation. (Inverse of `self.matrix2param_`.)

        For the sake of frugal computing, the `reduce_` option is introduced
        such that if True, this method returns `M * M_old^\dagger`,
        where `M_old` is the matrix constructed with self.sorted_phase.
        """
        phase, logJ_p2f = self.param2phase_(param)
        matrix, logJ_p2m = self.phase2matrix_(phase, reduce_=reduce_)
        return matrix, logJ_p2f + logJ_p2m

    @staticmethod
    def calc_log_conjugacy_vol(eigvals):
        """Return log of conjugacy volume up to an additive constant."""
        sumlogabs2 = lambda x: 2 * torch.sum(torch.log(torch.abs(x)), dim=-1)
        log_vol = torch.zeros(eigvals.shape[:-1], device=eigvals.device)
        for k in range(eigvals.shape[-1] - 1):
            log_vol += sumlogabs2(eigvals[..., k:k+1] - eigvals[..., k+1:])
        return log_vol.unsqueeze(-1)  # unsqueeze to keep dimensions the same

    @staticmethod
    def calc_conjugacy_vol(eigvals):
        """Return conjugacy volume up to a multiplacative constant."""
        prodabs2 = lambda x: torch.prod(torch.abs(x)**2, dim=-1)
        vol = torch.ones(eigvals.shape[:-1], device=eigvals.device)
        for k in range(eigvals.shape[-1] - 1):
            vol *= prodabs2(eigvals[..., k:k+1] - eigvals[..., k+1:])
        return vol.unsqueeze(-1)  # unsqueeze to keep dimensions the same

    def matrix2eigang_(self, *args, **kwargs):
        return self.matrix2phase_(*args, **kwargs)

    def eigang2matrix_(self, *args, **kwargs):
        return self.phase2matrix_(*args, **kwargs)

    def eigang2param_(self, *args, **kwargs):
        return self.phase2param_(*args, **kwargs)

    def param2eigang_(self, *args, **kwargs):
        return self.param2phase_(*args, **kwargs)

    def set_eigangs(self, eigangs):
        self.phase = eigangs

    def set_eigvecs(self, eigvecs):
        self.eigvecs = eigvecs

    @property
    def eigangs(self):  # eigen-angles; alias for phase
        return self.phase


# =============================================================================
class SUnMatrixParametrizer(UnMatrixParametrizer):

    def phase2param_(self, phase):
        self.order = ModalOrder(self.eigvecs)  # see order.sorted_ind
        sorted_phase = self.order.sort(phase)
        return self.sortedphase2param_(sorted_phase)

    def param2phase_(self, param):
        phase, logJ = self.param2sortedphase_(param)
        phase = self.order.revert(phase)  # revert the "sort" operation
        return phase, logJ

    @staticmethod
    def sortedphase2param_(sorted_phase):
        n = sorted_phase.shape[-1]
        return sorted_phase.split((1, n-1), dim=-1)[1], 0  # logJ = 0

    @staticmethod
    def param2sortedphase_(param):
        phase0 = -torch.sum(param, dim=-1).unsqueeze(-1)
        return torch.cat((phase0, param), dim=-1), 0  # logJ = 0


# =============================================================================
class SU2MatrixParametrizer(UnMatrixParametrizer):
    """Special case of SUnMatrixParametrizer with simpler methods."""

    def phase2param_(self, phase):
        self.order = ZeroSumOrder(phase)  # see order.(sorted_val & sorted_ind)
        return self.sortedphase2param_(self.order.sorted_val)

    def param2phase_(self, param):
        phase, logJ = self.param2sortedphase_(param)
        phase = self.order.revert(phase)  # revert the "sort" operation
        return phase, logJ

    @staticmethod
    def sortedphase2param_(sorted_phase):
        """Return parameters that vary between 0 and 1."""
        logJ = 0  # logJ = -np.log(pi) x N, but suppress the additive constant
        return sorted_phase[..., 1:] / pi, logJ

    @staticmethod
    def param2sortedphase_(param):
        """Inverse of sortedphase2param_()"""
        logJ = 0  # logJ = np.log(pi) x N, but suppress the additive constant
        return torch.cat((-param * pi, param * pi), dim=-1), logJ


# =============================================================================
class SU3MatrixParametrizer(UnMatrixParametrizer):
    """Special case of SUnMatrixParametrizer with simpler methods."""

    def phase2param_(self, phase):
        self.order = ZeroSumOrder(phase)  # see order.(sorted_val & sorted_ind)
        return self.sortedphase2param_(self.order.sorted_val)

    def param2phase_(self, param):
        phase, logJ = self.param2sortedphase_(param)
        phase = self.order.revert(phase)  # revert the "sort" operation
        return phase, logJ

    @staticmethod
    def sortedphase2param_(sorted_phase):
        r"""Return parameters that vary between 0 and 1.

        More precisely, return :math:`(w, r)` as defined below

        .. math::

            w = \theta \cos (\phi) / \pi  \in [0, 1] \\
            r = (1 + \tan (\phi) \sqrt{3}) / 2  \in [0, 1]

        Note that :math:`r = \sin(\phi + \pi/6) / \cos(\phi)`

        To convert w to \theta one can simply multiply it with
        :math:`\sqrt{1 + 4/3 (r - 1/2)^2}`.
        """
        x, y, z = sorted_phase.split((1, 1, 1), dim=-1)  # x <= y <= z
        w = (z - x) / (2 * pi)  # w \in [0, 1]
        w[w == 0] = 1e-16
        r = (y / w) * (3/4/pi) + 1/2  # r \in [0, 1]
        logJ = -sum_density(torch.log(w))
        # c = np.log(8/3 * pi**2), additive constant to log(w), but we drop it
        return torch.cat((w, r), dim=-1), logJ

    @staticmethod
    def param2sortedphase_(param):
        """Inverse of sortedphase2param_()"""
        w, r = param.split(1, dim=-1)
        y = w * (r - 1/2) / (3/4/pi)
        z = -y / 2 + w * pi
        x = -y / 2 - w * pi
        logJ = sum_density(torch.log(w))
        # c = np.log(8/3 * pi**2), additive constant to log(w), but we drop it
        return torch.cat((x, y, z), dim=-1), logJ


# =============================================================================
class U1Parametrizer:
    """Properties and methods are chosen to be consistent with SU(n)."""

    def matrix2param_(self, u1, **kwargs):
        """Return angle of eigenvalues and logJ of transformation."""
        phase = torch.angle(u1)  # in (-pi, pi]
        phase = phase.unsqueeze(-1)  # to be consistent with SU(n)
        self.phase = phase  # save for later use in param2matrix_
        param = phase / (2*pi) + 1/2
        logJ = 0  # logJ = -np.log(2 pi)xN, but suppress the additive constant
        return param, logJ

    def param2matrix_(self, param, reduce_=False):
        phase = (param - 1/2) * 2 * pi
        if reduce_:
            phase -= self.phase
        logJ = 0  # logJ = np.log(2 pi)xN, but suppress the additive constant
        return torch.exp(1j * phase).squeeze(-1), logJ


# =============================================================================
def sum_density(x):
    return torch.sum(x, dim=list(range(1, x.dim())))
