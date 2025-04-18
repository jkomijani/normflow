# Copyright (c) 2021-2025 Javad Komijani

# Components of this module are copied from '..matrix_handles.matrix_handle.py'

"""This module has utilities for eigenvalue decomposition."""

import torch

try:
    from lattice_ml.linalg import eigh
    from lattice_ml.linalg import eigu
    from lattice_ml.linalg import inverse_eign
except:
    from torch.linalg import eigh
    from torch.linalg import eig as eigu
    inverse_eign = lambda u, v: v @ (u.unsqueeze(-1) * v.adjoint())


# =============================================================================
class Eigd_:  # pylint: disable=invalid-name
    """
    A class for performing eigenvalue decomposition on diagonalizable matrices.

    In addition to calculating the eigenvalues and eigenvectors, this class
    also computes the negative logarithm of the conjugacy volume of the
    eigenvalues. The conjugacy volume is interpreted as the inverse of the
    Jacobian of the transformation, which is particularly useful in methods
    such as normalizing flows.

    The *conjugacy volume* refers to the "volume" of the space of matrices
    that are similar (conjugate) to a given matrix. All matrices in this space
    share the same eigenvalues, but differ in their eigenvectors. For
    diagonalizable matrices, the conjugacy volume is fully determined by the
    eigenvalues.

    Example:
    --------
    eigd_ = Eigd_()
    (eigvals, eigvecs), log_jacobian = eigd_(matrix)
    """

    eig = torch.linalg.eig  # Default to PyTorch's eigen decomposition function

    def __call__(self, matrix):
        """
        Computes the eigenvalues and eigenvectors of the given matrix and the
        logarithm of the Jacobian of the transformation.

        Parameters:
        -----------
        matrix : torch.Tensor
            A square matrix for which eigenvalues and eigenvectors are to be
            computed. The matrix is assumed to be diagonalizable, but no error
            is raised for non-diagonalizable matrices.

        Returns:
        --------
        tuple : ((torch.Tensor, torch.Tensor), torch.Tensor)
            A tuple containing:
            - A tuple of tensors: (eigenvalues, eigenvectors).
            - The logarithm of the Jacobian of the transformation.
        """

        # Compute eigenvalues & eigenvectors using underlying `eig` function
        eigvals, eigvecs = self.eig(matrix)

        # Compute the logarithm of the Jacobian (summed over non-batched axes)
        log_jacobian = -sum_density(calc_log_conjugacy_vol(eigvals))

        # Return eigenvalues, eigenvectors, and the log-jacobian term
        return (eigvals, eigvecs), log_jacobian


class Eign_(Eigd_):  # pylint: disable=invalid-name
    """A (dummy) subclass of `Eigd_` for normal matrices."""
    pass


class Eigh_(Eigd_):  # pylint: disable=invalid-name
    """A subclass of `Eigd_` specialized for Hermitian matrices."""
    eig = eigh


class Eigu_(Eigd_):  # pylint: disable=invalid-name
    """A subclass of `Eigd_` specialized for unitary matrices."""
    eig = eigu


class InverseEign_:  # pylint: disable=invalid-name
    """
    A class for matrix recomposition from eigenvalues and eigencectors, where
    the matrix of the eigenvectors is unitary. (Therefore, the constructed
    matrix is *normal*.)

    In addition to constructing the matrix, this class also computes the
    logarithm of the conjugacy volume of the eigenvalues. The conjugacy volume
    is interpreted as the Jacobian of the transformation.

    The *conjugacy volume* refers to the "volume" of the space of matrices
    that are similar (conjugate) to a given matrix. All matrices in this space
    share the same eigenvalues, but differ in their eigenvectors. For normal
    (and in general all diagonalizable) matrices, the conjugacy volume is fully
    determined by the eigenvalues.

    Example:
    --------
    inverse_eign_ = InverseEign_()
    matrix = inverse_eign_(eigvals, eigvecs)
    """

    def __call__(self, eigvals, eigvecs):

        # Construct the matrix with inverse_eign, valid for normal matrices.
        matrix = inverse_eign(eigvals, eigvecs)

        # Compute the logarithm of the Jacobian (summed over non-batched axes)
        log_jacobian = sum_density(calc_log_conjugacy_vol(eigvals))

        # Return the constructed matrix and the log-jacobian term
        return matrix, log_jacobian


# =============================================================================
# We now make instances of the above classes

eigh_ = Eigh_()  # for Hermitian matrices

eigu_ = Eigu_()  # for unitary matrices

inverse_eign_ = InverseEign_()  # for normal (including Hermitian & unitray)


# =============================================================================
def sum_density(x):
    return torch.sum(x, dim=list(range(1, x.dim())))


def calc_log_conjugacy_vol(eigvals):
    r"""
    Calculate the log of the conjugacy volume up to an additive constant.

    This function computes a volume measure related to the conjugacy class of a
    matrix based on its eigenvalues. The conjugacy volume is:

    .. math::

        \prod_{k < l} |\lambda_k - \lambda_l|^2

    where :math:`\lambda_k` are the eigenvalues.

    Parameters:
    -----------
    eigvals : torch.Tensor
        A tensor of eigenvalues with shape (..., n),  where `n` is the number
        of eigenvalues, and `...` represents any leading dimensions.

    Returns:
    --------
    torch.Tensor
        A tensor with the log of the conjugacy volume, having the same shape as
        `eigvals` with an added singleton dimension at the end, i.e., shape
        (..., 1).

    Example:
        >>> eigvals = torch.tensor([[0.0, 0.5, 1.0], [0, 1, 2]])
        >>> calc_log_conjugacy_vol(eigvals)
        tensor([[-2.7726], [ 1.3863]])
    """

    sumlogabs2 = lambda x: 2 * torch.sum(torch.log(torch.abs(x)), dim=-1)

    log_vol = torch.zeros(eigvals.shape[:-1], device=eigvals.device)

    for k in range(eigvals.shape[-1] - 1):
        log_vol += sumlogabs2(eigvals[..., k:k+1] - eigvals[..., k+1:])

    return log_vol.unsqueeze(-1)  # unsqueeze to keep dimensions the same


def calc_conjugacy_vol(eigvals):
    r"""
    Calculate the conjugacy volume up to a multiplacative constant.

    This function computes a volume measure related to the conjugacy class of a
    matrix based on its eigenvalues. The conjugacy volume is:

    .. math::

        \prod_{k < l} |\lambda_k - \lambda_l|^2

    where :math:`\lambda_k` are the eigenvalues.

    Parameters:
    -----------
    eigvals : torch.Tensor
        A tensor of eigenvalues with shape (..., n),  where `n` is the number
        of eigenvalues, and `...` represents any leading dimensions.

    Returns:
    --------
    torch.Tensor
        A tensor with the conjugacy volume, having the same shape as `eigvals`
        with an added singleton dimension at the end, i.e., shape (..., 1).

    Example:
        >>> eigvals = torch.tensor([[0.0, 0.5, 1.0], [0, 1, 2]])
        >>> calc_conjugacy_vol(eigvals)
        tensor([[0.0625], [4.0000]])
    """

    prodabs2 = lambda x: torch.prod(torch.abs(x)**2, dim=-1)

    vol = torch.ones(eigvals.shape[:-1], device=eigvals.device)

    for k in range(eigvals.shape[-1] - 1):
        vol *= prodabs2(eigvals[..., k:k+1] - eigvals[..., k+1:])

    return vol.unsqueeze(-1)  # unsqueeze to keep dimensions the same
