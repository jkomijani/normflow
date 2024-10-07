# Copyright (c) 2024 Javad Komijani


import torch
from typing import Union, Callable

from .._core import Module_
from .._core import ModuleList_
from .modules_ import Pade32_, Affine_


class MultiPlanarFlow_(Module_):
    """Performs a transformation as

    .. math::

        Y = W \Sigma^{-1} h(W^\dagger X) + P X.

    Here `X` and `Y` are vectors of size `n`, `W` is a matrix of size `[n, m]`,
    where `m` specifies the number of (hidden) channels of transformation,
    `\Sigma = W^\dagger W`, `P` is a projection matrix perpendicular to `W`,
    and `h` is an invertible nonlinearity.

    Rezende and Mohamed [arXiv:1505.05770] introduced planar flows, which take
    the form ``Y = X + U h(W^\dagger X)``, with ``m = 1``. The transformation
    used here is a genarlization of planar flow to ``m`` channels and with a
    projection matrix.

    Parameters
    ----------
    n: int
        lenght of the outermost axix of input
    m: int
        number of (hidden) channels
    net_: Union[Callable, None], optional
        a module that transforms the projected data and also returns `logj`
        of transformation. (It is `h` in the above expression.) If not provided
        as input, the default case would be used, which is
        ``net_ = Pade32_(**kwargs) o Affine_(**kwargs)``, where ``**kwargs``
        provides information about number of channels: If `m == 1`, no channels
        are introduced. Otherwise, the number of channels is set to `m`.
    """
    def __init__(self,
                 n: int,
                 m: int,
                 net_: Union[Callable, None] = None,
                 set_param2zero: bool = True
                ):

        assert n > 1 and m > 0, "Appropriate only for n > 1 & m > 0"

        def make_parameters(*shape):
            var = torch.randn(*shape)
            return torch.nn.Parameter(var / torch.norm(var))

        super().__init__()

        self.n_dim = n
        self.m_dim = m

        self.w_mat = make_parameters(1, n, m)
        self.eye = torch.eye(n).unsqueeze(0)

        if net_ is None:
            # if m > 1 set n_channels = m & channels axis = 1
            kws = dict(channels_axis = None if m==1 else 1, n_channels=m)
            net_ = ModuleList_([Pade32_(**kws), Affine_(**kws)])

        self.net_ = net_

        if set_param2zero:
            self.set_param2zero()

    def forward(self, x, log0=0):

        assert x.shape[-1] == self.n_dim, "inconsistent input"

        x_shape = x.shape  # keep for later use
        x = x.reshape(-1, self.n_dim, 1)

        w_mat = self.eye[..., :self.m_dim] + self.w_mat
        ws_mat = w_mat @ torch.linalg.inv(w_mat.adjoint() @ w_mat)

        y_parallel, logj = self.net_.forward(w_mat.adjoint() @ x)

        y = ws_mat @ y_parallel + (self.eye - ws_mat @ w_mat.adjoint()) @ x

        return y.reshape(*x_shape), log0 + logj

    def reverse(self, y, log0=0):

        assert y.shape[-1] == self.n_dim, "inconsistent input"

        y_shape = y.shape  # keep for later use
        y = y.reshape(-1, self.n_dim, 1)

        w_mat = self.eye[..., :self.m_dim] + self.w_mat
        ws_mat = w_mat @ torch.linalg.inv(w_mat.adjoint() @ w_mat)

        x_parallel, logj = self.net_.reverse(w_mat.adjoint() @ y)

        x = ws_mat @ x_parallel + (self.eye - ws_mat @ w_mat.adjoint()) @ y

        return x.reshape(*y_shape), log0 + logj
