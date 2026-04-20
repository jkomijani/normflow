# Copyright (c) 2021-2025 Javad Komijani

"""This module has extensions to the linalg packages in torch."""


import torch

from lattice_ml.linalg import svd as lattice_ml_svd


from .eig_decomposition_ import eigh
from .eig_decomposition_ import eigu
from .eig_decomposition_ import inverse_eign

from .eig_decomposition_ import eigh_
from .eig_decomposition_ import eigu_
from .eig_decomposition_ import inverse_eigh_
from .eig_decomposition_ import inverse_eign_


from .qr_decomposition import haar_qr, haar_sqr

from .euler_angles import su2_to_euler_angles
from .euler_angles import euler_angles_to_su2


def compute_svd(matrix: torch.Tensor):
    """
    Compute the singular value decomposition (SVD) using the PyTorch backend.

    This is a wrapper around `lattice_ml.linalg.svd` that explicitly selects
    `backend="torch"`. The result is returned as an instance of `SVDResult`,
    ensuring compatibility with the automatic differentiation (AD) setup
    defined there.

    Note:
        - The native PyTorch AD for `torch.linalg.svd` is not reliable for this
          use case (e.g., may produce incorrect or unstable gradients), hence
          the use of the custom AD defined in `SVDResult`.
        - In single precision, `torch.linalg.svd` is slightly more accurate
          numerically than the custom implementation in `lattice_ml`.

    Args:
        matrix (torch.Tensor): Input tensor of shape (..., n, n).

    Returns:
        SVDResult: Structured SVD result.
    """
    return lattice_ml_svd(matrix, backend="torch")
