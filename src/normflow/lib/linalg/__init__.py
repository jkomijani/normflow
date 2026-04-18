# Copyright (c) 2021-2025 Javad Komijani

"""This module has extensions to the linalg packages in torch."""


import torch

from lattice_ml.linalg import svd as compute_svd


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
