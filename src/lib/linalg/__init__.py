# Copyright (c) 2021-2024 Javad Komijani

"""This module has extensions to the linalg packages in torch."""


import torch


try:
    from torch_linalg_ext import svd
except:
    from torch.linalg import svd


from .eig_decomposition_ import eigh
from .eig_decomposition_ import eigu
from .eig_decomposition_ import inverse_eign

from .eig_decomposition_ import eigh_
from .eig_decomposition_ import eigu_
from .eig_decomposition_ import inverse_eign_


from .qr_decomposition import haar_qr, haar_sqr

from .euler_angles import su2_to_euler_angles
from .euler_angles import euler_angles_to_su2


class AttributeDict4SVD:
    """For accessing a dict key like an attribute."""

    def __init__(self, **dict_):
        self.__dict__.update(**dict_)

    def __repr__(self):
        str_ = "svd:\n"
        for key, value in self.__dict__.items():
            str_ += f"{key}={value}\n"
        return str_


def special_svd(matrix):
    """Return a new svd object, in which U is scaled by a phase, and called sU,
    where s stands for special, such that sU @ Vh is special unitary.
    """
    svd_ = svd(matrix)
    rdet_angle = torch.angle(torch.det(matrix)) / svd_.U.shape[-1]  # r: rooted
    phase_factor = torch.exp(-1j * rdet_angle.reshape(*rdet_angle.shape, 1, 1))
    s_uvh = (svd_.U @ svd_.Vh) * phase_factor  # s stands for special
    # s_u = svd_.U * torch.exp(-1j * phase_factor)
    sigma_matrix = svd_.Vh.adjoint() @ (svd_.S.unsqueeze(-1) * svd_.Vh)
    return AttributeDict4SVD(
        U=svd_.U, S=svd_.S, Vh=svd_.Vh, rdet_angle=rdet_angle, sUVh=s_uvh,
        Sigma=sigma_matrix
        )
