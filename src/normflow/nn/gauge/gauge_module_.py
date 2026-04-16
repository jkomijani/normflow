# Copyright (c) 2021-2026 Javad Komijani

"""This module contains new neural networks for transforming gauge fields.

The classes defined here are children of MatrixModule_ (and in turn Module_),
and the trailing underscore implies that the associated forward and reverse
methods handle the Jacobians of the transformation.
"""

# pylint: disable=invalid-name, relative-beyond-top-level

import torch

from .._core import Module_
from .._core import ModuleList_
from ..matrix.matrix_module_ import MatrixModule_
from ..matrix.stapled_matrix_module_ import StapledMatrixModule_


# =============================================================================
class GaugeModuleList_(ModuleList_):
    """
    Container for a sequence of `GaugeModule_` transformations.

    Applies gauge updates sequentially along a chosen link axis.

    Attributes
    ----------
    unbind_link_axis : bool
        If True, the input is split along `link_axis` into a list of tensors
        before applying the modules, and stacked back afterward. This lets each
        `GaugeModule_` operate on a single link direction (internally assuming
        link_axis = 0).
        If False, the full tensor is passed directly to each module.
    """

    unbind_link_axis = True  # internally switch link_axis to 0

    def __init__(
        self,
        list_of_gauge_modules_,
        sites_before_link: bool = True
    ):
        """
        Initialize the module list.

        Parameters
        ----------
        list_of_gauge_modules_ : list[GaugeModule_]
            Sequence of gauge update modules.
        sites_before_link : bool, optional
            If True, site dimensions precede the link axis in `x`;
            otherwise the link axis comes earlier.
        """
        super().__init__(list_of_gauge_modules_)
        self.sites_before_link = sites_before_link

    @property
    def link_axis(self):
        """
        Axis corresponding to link directions in the input tensor.
        """
        return -3 if self.sites_before_link else 1

    def forward(self, x, log0=0):
        """
        Apply forward transformations sequentially.

        Splits and recombines along `link_axis` if `unbind_link_axis` is
        enabled. Returns updated `x` and accumulated log-Jacobian.
        """
        if self.unbind_link_axis:
            x = list(torch.unbind(x, self.link_axis))
            x, log0 = super().forward(x, log0)
            return torch.stack(x, dim=self.link_axis), log0
        else:
            return super().forward(x.clone(), log0)

    def reverse(self, x, log0=0):
        """
        Apply inverse transformations in reverse order.

        Mirrors `forward`, including optional splitting along `link_axis`.
        Returns updated `x` and accumulated log-Jacobian.
        """
        if self.unbind_link_axis:
            x = list(torch.unbind(x, self.link_axis))
            x, log0 = super().reverse(x, log0)
            return torch.stack(x, dim=self.link_axis), log0
        else:
            return super().reverse(x.clone(), log0)

    def hack(self, x, log0=0):
        """Similar to the forward method, except that returns the output of
        middle blocks too; useful for examining effects of each block.
        """
        stack = [(x, log0)]

        if self.unbind_link_axis:
            for net_ in self:
                x = list(torch.unbind(x, self.link_axis))
                x, log0 = net_(x, log0)
                x = torch.stack(x, dim=self.link_axis)
                stack.append([x, log0])
            return stack
        else:
            return None


# =============================================================================
class GaugeModule_(Module_):
    """
    Gauge-equivariant link update module.

    Applies an invertible transformation to links in direction `mu` using
    staple information in planes defined by `nu_list`. The update is performed
    via spectral decomposition (eigen-angles/vectors) and optional neural
    networks acting on parameters, eigen-angles, and eigenvectors.

    Parameters
    ----------
    mu : int
        specifies the direction of links that are going to be changed.

    nu_list : list of int
        (in combination w/ mu) specifies the plane of staples to be calculated.

    staple_handle: class instance
        to calculate staples and use them.

    matrix_handle: class instance
        to handle parametrization of the matrices.

    param_net_: instance of Module_ or ModuleList_
        a core network to change a set of parameters corresponding to the
        stapled links.

    eigangs_net_: instance of Module_ or ModuleList_ (optional)
        a network to change the eigen-anlges of the stapled links.
        (Default is None.)

    eigvals_net_: instance of Module_ or ModuleList_ (optional)
        a network to change the eigen-vectors of the stapled links.
        (Default is None.)

    Notes
    -----
    Transform is invertible if all networks use compatible masks.
    """

    unbounded_link_axis = True  # "link_axis" of inputs is supposed to be 0
    sites_before_link = True  # It is irrelavant if unbounded_link_axis is True

    def __init__(
        self,
        mu,
        nu_list,
        staples_handle,
        matrix_handle,
        param_net_,
        dual_param_net_=None,
        eigangs_net_=None,
        eigvecs_net_=None,
        staples_kwargs=None
    ):
        super().__init__()
        self.mu = mu
        self.nu_list = nu_list

        self.param_net_ = param_net_
        self.dual_param_net_ = dual_param_net_
        self.eigangs_net_ = eigangs_net_
        self.eigvecs_net_ = eigvecs_net_

        self.matrix_handle = matrix_handle

        self.staples_handle = staples_handle
        self.staples_kwargs = staples_kwargs or {}

        self.link_axis = self._resolve_link_axis()
        # Change the link_axis in staples_handle accordingly
        self.staples_handle.link_axis = self.link_axis

    # -------------------------------------------------------------------------
    # public API
    # -------------------------------------------------------------------------

    def forward(self, x, log0=0):
        """
        Apply forward link transformation.

        Computes staples, transforms stapled links, and updates `x`.
        Returns updated `x` and accumulated log-Jacobian.
        """
        return self._update_links(x, log0, reverse=False)

    def reverse(self, x, log0=0):
        """
        Apply inverse link transformation.

        Reverses the forward update using inverse network operations.
        Returns updated `x` and accumulated log-Jacobian.
        """
        return self._update_links(x, log0, reverse=True)

    # -------------------------------------------------------------------------
    # core pipeline
    # -------------------------------------------------------------------------

    def _update_links(self, x, log0, reverse):
        """
        Apply a forward or inverse update to links in direction `mu`.

        The update proceeds by:
        1. Computing staples around each link.
        2. Forming the "stapled link" (slink) by attaching a projected staple
           contribution via SVD (see `staples_handle.staple`).
        3. Applying an invertible transformation to the slink.
        4. Converting the transformed slink into a rotation and pushing it
           back onto the original link.
        5. Writing the updated links back into `x`.

        Parameters
        ----------
        x : tensor-like
            Input gauge field.
        log0 : scalar
            Initial log-Jacobian accumulator.
        reverse : bool
            If True, applies the inverse transformation.

        Returns
        -------
        x : tensor-like
            Updated gauge field.
        logj : scalar
            Accumulated log-Jacobian.
        """
        staples_ctx = self._compute_staples(x)  # staple context: data &helpers

        x_mu = self._get_x_mu(x)
        slink = self._build_slink(x_mu, staples_ctx)

        if reverse:
            new_slink, logj = self._apply_transform_reverse(slink, staples_ctx)
        else:
            new_slink, logj = self._apply_transform_forward(slink, staples_ctx)

        x_mu = self._push_back(x_mu, slink, new_slink, staples_ctx)
        x = self._set_x_mu(x, x_mu)

        return x, log0 + logj

    def _compute_staples(self, x):
        """Return staple context (data and helpers) for link update."""
        return self.staples_handle.calc_staples(
            x, mu=self.mu, nu_list=self.nu_list, **self.staples_kwargs
        )

    def _build_slink(self, x_mu, staples_ctx):
        return self.staples_handle.staple(x_mu, staples_ctx)

    def _push_back(self, x_mu, slink, new_slink, staples_ctx):
        return self.staples_handle.push2link(
            x_mu, new_slink @ slink.adjoint(), staples_ctx,
        )

    def _apply_transform_forward(self, slink, staples_object):
        """
        Transform a stapled link via spectral decomposition.

        Applies parameter, eigen-angle, and eigenvector networks.
        Returns transformed link and log-Jacobian.
        """
        # slink: stapled link
        # ======
        # Spectral decomposition of the input matrix
        eigangs, logJ_mat2ang = self.matrix_handle.matrix2eigang_(slink)
        eigvecs = self.matrix_handle.eigvecs
        logJ = logJ_mat2ang

        # Part 1: parametrize the eigenangles and transform the parameters
        if self.param_net_ is not None:
            param, logJ_ang2par = self.matrix_handle.eigang2param_(eigangs)
            param, logJ_par2par = self.param_net_(param)
            if self.dual_param_net_ is not None:
                param, logJ_par2par = self.dual_param_net_(
                    param,
                    staples_object.get_dual_param(eigvecs),
                    log0=logJ_par2par
                )
            eigangs, logJ_par2ang = self.matrix_handle.param2eigang_(param)
            logJ += (logJ_ang2par + logJ_par2par + logJ_par2ang)

        # ======
        # Part 2: transform the eigenvalues directly
        if self.eigangs_net_ is not None:
            eigangs, logJ_ang2ang = self.eigangs_net_(
                eigangs, eigvecs=eigvecs, staples_object=staples_object
            )
            logJ += logJ_ang2ang

        # ======
        # Part 3: transform the eigenvectors
        if self.eigvecs_net_ is not None:
            eigvecs, logJ_vec2vec = self.eigvecs_net_(
                eigvecs, eigangs=eigangs, staples_object=staples_object
            )
            logJ += logJ_vec2vec
            self.matrix_handle.set_eigvecs(eigvecs)  # save the new eigvecs

        # ======
        # Finally, put all pieces together
        new_slink, logJ_ang2mat = self.matrix_handle.eigang2matrix_(eigangs)
        logJ += logJ_ang2mat

        return new_slink, logJ

    def _apply_transform_reverse(self, slink, staples_object):
        """
        Inverse transform of a stapled link.

        Reverses eigenvector, eigen-angle, and parameter transforms.
        Returns transformed link and log-Jacobian.
        """
        # slink: stapled link
        # ======
        # Part 0: Spectral decomposition of the input matrix
        eigangs, logJ_mat2ang = self.matrix_handle.matrix2eigang_(slink)
        eigvecs = self.matrix_handle.eigvecs
        logJ = logJ_mat2ang

        # ======
        # Part inverse-3: inverse-transform the eigenvectors
        if self.eigvecs_net_ is not None:
            eigvecs, logJ_vec2vec = self.eigvecs_net_.reverse(
                eigvecs, eigangs=eigangs, staples_object=staples_object
            )
            logJ += logJ_vec2vec
            self.matrix_handle.set_eigvecs(eigvecs)  # save the new eigvecs

        # ======
        # Part inverse-2: inverse-transform the eigenvalues directly
        if self.eigangs_net_ is not None:
            eigangs, logJ_ang2ang = self.eigangs_net_.reverse(
                eigangs, eigvecs=eigvecs, staples_object=staples_object
            )
            logJ += logJ_ang2ang

        # Part inverse-1: inverse-transform the parameters
        if self.param_net_ is not None:
            param, logJ_ang2par = self.matrix_handle.eigang2param_(eigangs)
            if self.dual_param_net_ is not None:
                param, logJ_par2par = self.dual_param_net_.reverse(
                    param, staples_object.get_dual_param(eigvecs)
                )
            else:
                logJ_par2par = 0
            log0 = logJ_par2par
            param, logJ_par2par = self.param_net_.reverse(param, log0=log0)
            eigangs, logJ_par2ang = self.matrix_handle.param2eigang_(param)
            logJ += (logJ_ang2par + logJ_par2par + logJ_par2ang)

        # ======
        # Finally, put all pieces together
        new_slink, logJ_ang2mat = self.matrix_handle.eigang2matrix_(eigangs)
        logJ += logJ_ang2mat

        return new_slink, logJ

    # ------------------------------------------------------------------
    # indexing helpers
    # ------------------------------------------------------------------

    def _resolve_link_axis(self):
        if self.unbounded_link_axis:
            return 0
        return -3 if self.sites_before_link else 1

    def _get_x_mu(self, x):
        """Extract links in direction `mu` from input tensor `x`."""
        if self.unbounded_link_axis:
            x_mu = x[self.mu]
        elif not self.sites_before_link:
            x_mu = x[:, self.mu]
        else:
            x_mu = x[..., self.mu, :, :]
        return x_mu

    def _set_x_mu(self, x, x_mu):
        """Set links in direction `mu` in `x` to `x_mu`."""
        if self.unbounded_link_axis:
            x[self.mu] = x_mu
        elif not self.sites_before_link:
            x[:, self.mu] = x_mu
        else:
            x[..., self.mu, :, :] = x_mu
        return x


# =============================================================================
class _GaugeModule_(MatrixModule_):
    """
    Parameters
    ----------
    param_net_: instance of Module_ or ModuleList_
        a core network to change a set of parameters corresponding to the
        stapled links as specified in the supper class `MatrixModule_`.

    mu : int
        specifies the direction of links that are going to be changed

    nu_list : list of int
        (in combination w/ mu) specifies the plane of staples to be calculated

    staple_handle: class instance
        to calculate staples and use them.

    matrix_handle: class instance
        to handle matrices as expected in the supper class `MatrixModule_`.
    """

    unbounded_link_axis = True

    def __init__(
        self, param_net_,
        *, mu, nu_list, staples_handle, matrix_handle,
        staples_coeff=None, label="gauge_"
    ):
        super().__init__(param_net_, matrix_handle=matrix_handle)
        self.mu = mu
        self.nu_list = nu_list
        self.staples_handle = staples_handle
        if self.unbounded_link_axis:
            self.staples_handle.link_axis = 0
        self.staples_coeff = staples_coeff
        self.label = label

    def forward(self, x, log0=0):
        if self.unbounded_link_axis:
            x_mu = x[self.mu]
        else:
            x_mu = x[:, self.mu]

        staples_object = self.staples_handle.calc_staples(
            x, mu=self.mu, nu_list=self.nu_list,
            staples_coeff=self.staples_coeff
        )

        # slink: stapled link
        slink = self.staples_handle.staple(x_mu, staples_object=staples_object)

        slink_rotation, logJ = super().forward(slink, log0=log0, reduce_=True)

        x_mu = self.staples_handle.push2link(
            x_mu, slink_rotation=slink_rotation, staples_object=staples_object
        )

        if self.unbounded_link_axis:
            x[self.mu] = x_mu
        else:
            x[:, self.mu] = x_mu

        return x, logJ

    def reverse(self, x, log0=0):
        if self.unbounded_link_axis:
            x_mu = x[self.mu]
        else:
            x_mu = x[:, self.mu]

        staples_object = self.staples_handle.calc_staples(
            x, mu=self.mu, nu_list=self.nu_list,
            staples_coeff=self.staples_coeff
        )

        slink = self.staples_handle.staple(x_mu, staples_object=staples_object)

        slink_rotation, logJ = super().reverse(slink, log0=log0, reduce_=True)

        x_mu = self.staples_handle.push2link(
            x_mu, slink_rotation=slink_rotation, staples_object=staples_object
        )

        if self.unbounded_link_axis:
            x[self.mu] = x_mu
        else:
            x[:, self.mu] = x_mu

        return x, logJ

    def _hack(self, x, forward=True, unbind_link_axis=True):
        """Similar to the forward method, but returns intermediate parts."""

        if unbind_link_axis:
            x = list(torch.unbind(x, 1))

        x_mu = x[self.mu]

        staples_object = self.staples_handle.calc_staples(
            x, mu=self.mu, nu_list=self.nu_list,
            staples_coeff=self.staples_coeff
        )
        slink = self.staples_handle.staple(x_mu, staples_object=staples_object)

        if forward:
            slink_rotation, logJ = super().forward(slink, reduce_=True)
        else:
            slink_rotation, logJ = super().reverse(slink, reduce_=True)

        stack = dict(
            x_mu_initial=x_mu,
            staples_object=staples_object,
            slink=slink,
            slink_rotation=slink_rotation,
            logJ=logJ,
            super_hack=super()._hack(slink, forward, reduce_=True)
        )

        x_mu = self.staples_handle.push2link(
            x_mu, slink_rotation=slink_rotation, staples_object=staples_object
        )
        stack["x_mu_final"] = x_mu

        return stack

    def transfer(self, **kwargs):
        return self.__class__(
                self.param_net_.transfer(**kwargs),
                mu=self.mu,
                nu_list=self.nu_list,
                staples_handle=self.staples_handle,
                matrix_handle=self.matrix_handle,
                label=self.label
                )


# =============================================================================
class _SVDGaugeModule_(StapledMatrixModule_):
    """
    Similar to GaugeModule_ but uses singular values of the staples for
    processing too.

    Parameters
    ----------
    dual_param_net_: instance of Module_ or ModuleList_
        a core network to change a set of parameters corresponding to the
        stapled links as specified in the supper class `StapledMatrixModule_`.

    param_net_: instance of Module_ or ModuleList_
        a core network to change a set of parameters corresponding to the
        stapled links as specefied in the supper class `StapledMatrixModule_`.

    mu : int
        specifies the direction of links that are going to be changed

    nu_list : list of int
        (in combination w/ mu) specifies the plane of staples to be calculated

    staple_handle: class instance
        to calculate staples and use them.

    matrix_handle: class instance
        to handle matrices as expected in the supper class `MatrixModule_`.

    IMPORTANT NOTE:
        in order to have an invertible forward method, the masks used in
        `param_net_` and `dual_param_net_` must be compatible.
    """

    unbounded_link_axis = True

    def __init__(
        self, dual_param_net_, param_net_,
        *, mu, nu_list, staples_handle, matrix_handle,
        staples_coeff=None, label="gauge_", **kwargs
    ):
        super().__init__(
            dual_param_net_, param_net_, matrix_handle=matrix_handle, **kwargs
            )
        self.mu = mu
        self.nu_list = nu_list
        self.staples_handle = staples_handle
        if self.unbounded_link_axis:
            self.staples_handle.link_axis = 0
        self.staples_coeff = staples_coeff
        self.label = label

    def forward(self, x, log0=0):
        if self.unbounded_link_axis:
            x_mu = x[self.mu]
        else:
            x_mu = x[:, self.mu]

        staples_object = self.staples_handle.calc_staples(
            x, mu=self.mu, nu_list=self.nu_list,
            staples_coeff=self.staples_coeff
        )

        # slink: stapled link
        slink = self.staples_handle.staple(x_mu, staples_object=staples_object)

        slink_rotation, logJ = super().forward(
            slink, log0=log0, singv=staples_object.singv, reduce_=True
        )

        x_mu = self.staples_handle.push2link(
            x_mu, slink_rotation=slink_rotation, staples_object=staples_object
        )

        if self.unbounded_link_axis:
            x[self.mu] = x_mu
        else:
            x[:, self.mu] = x_mu

        return x, logJ

    def reverse(self, x, log0=0):
        if self.unbounded_link_axis:
            x_mu = x[self.mu]
        else:
            x_mu = x[:, self.mu]

        staples_object = self.staples_handle.calc_staples(
            x, mu=self.mu, nu_list=self.nu_list,
            staples_coeff=self.staples_coeff
        )

        slink = self.staples_handle.staple(x_mu, staples_object=staples_object)

        slink_rotation, logJ = super().reverse(
            slink, log0=log0, singv=staples_object.singv, reduce_=True
        )

        x_mu = self.staples_handle.push2link(
            x_mu, slink_rotation=slink_rotation, staples_object=staples_object
        )

        if self.unbounded_link_axis:
            x[self.mu] = x_mu
        else:
            x[:, self.mu] = x_mu

        return x, logJ

    def _hack(self, x, forward=True, unbind_link_axis=True):
        """Similar to the forward method, but returns intermediate parts."""

        if unbind_link_axis:
            x = list(torch.unbind(x, 1))

        x_mu = x[self.mu]

        staples_object = self.staples_handle.calc_staples(
            x, mu=self.mu, nu_list=self.nu_list,
            staples_coeff=self.staples_coeff
        )
        slink = self.staples_handle.staple(x_mu, staples_object=staples_object)

        if forward:
            slink_rotation, logJ = super().forward(
                slink, singv=staples_object.singv, reduce_=True
            )
        else:
            slink_rotation, logJ = super().reverse(
                slink, singv=staples_object.singv, reduce_=True
            )

        stack = dict(
            x_mu_initial=x_mu,
            staples_object=staples_object,
            slink=slink,
            slink_rotation=slink_rotation,
            logJ=logJ,
            super_hack=super()._hack(
               slink, singv=staples_object.singv, forward=forward, reduce_=True
            )
        )

        x_mu = self.staples_handle.push2link(
            x_mu, slink_rotation=slink_rotation, staples_object=staples_object
        )
        stack["x_mu_final"] = x_mu

        return stack


# =============================================================================
class PolyakovGaugeModule_(MatrixModule_):

    unbounded_link_axis = True

    def __init__(
        self, param_net_, *, mu, nu_list, staples_handle, matrix_handle, parity
    ):
        super().__init__(param_net_, matrix_handle=matrix_handle)
        self.mu = mu
        self.nu_list = nu_list
        self.staples_handle = staples_handle
        if self.unbounded_link_axis:
            self.staples_handle.link_axis = 0
        self.parity = parity

    def forward(self, x, log0=0):

        mu, x_mu = self.mu, x[self.mu]
        loop_dim = 1 + mu  # 1 is for the batch axis

        polyakov_loop = matrix_product(torch.unbind(x_mu, dim=loop_dim))

        staples = self.staples_handle.calc_staples(
            x, mu=mu, nu_list=self.nu_list
        )

        polyakov_staples = matrix_product(
            torch.unbind(staples, dim=loop_dim), right_product=False
        )

        # slink: stapled link
        sploop, svd_ = self.staples_handle.staple(
            polyakov_loop, staples=polyakov_staples
        )

        rotation, logJ = super().forward(sploop, reduce_=True)
        rotation = select_embed(x_mu.shape, rotation, loop_dim, 0)
        svd_.sU = select_embed(x_mu.shape, svd_.sU, loop_dim, 0)
        svd_.Vh = select_embed(x_mu.shape, svd_.Vh, loop_dim, 0)

        x[mu] = self.staples_handle.push2link(
            x_mu, slink_rotation=rotation, svd_=svd_
        )

        return x, log0 + logJ

    def reverse(self, x, log0=0):

        mu, x_mu = self.mu, x[self.mu]
        loop_dim = 1 + mu  # 1 is for the batch axis

        polyakov_loop = matrix_product(torch.unbind(x_mu, dim=loop_dim))

        staples = self.staples_handle.calc_staples(
            x, mu=self.mu, nu_list=self.nu_list
        )

        polyakov_staples = matrix_product(
            torch.unbind(staples, dim=loop_dim), right_product=False
        )

        # slink: stapled link
        sploop, svd_ = self.staples_handle.staple(
            polyakov_loop, staples=polyakov_staples
        )

        rotation, logJ = super().reverse(sploop, reduce_=True)
        rotation = select_embed(x_mu.shape, rotation, loop_dim, 0)
        svd_.sU = select_embed(x_mu.shape, svd_.sU, loop_dim, 0)
        svd_.Vh = select_embed(x_mu.shape, svd_.Vh, loop_dim, 0)

        x[mu] = self.staples_handle.push2link(
            x_mu, slink_rotation=rotation, svd_=svd_
        )

        return x, log0 + logJ

    def _forward(self, x, log0=0):

        mu, x_mu = self.mu, x[self.mu]
        dim = 1 + mu  # 1 is for the batch axis

        polyakov_loop = matrix_product(torch.unbind(x_mu, dim=dim))
        rotation, logJ = super().forward(polyakov_loop, reduce_=True)
        rotation = select_embed(x_mu.shape, rotation, dim, 0)

        x[mu] = rotation @ x_mu
        return x, log0 + logJ

    def _reverse(self, x, log0=0):

        mu, x_mu = self.mu, x[self.mu]
        dim = 1 + mu  # 1 is for the batch axis

        polyakov_loop = matrix_product(torch.unbind(x_mu, dim=dim))
        rotation, logJ = super().reverse(polyakov_loop, reduce_=True)
        rotation = select_embed(x_mu.shape, rotation, dim, 0)

        x[mu] = rotation @ x_mu
        return x, log0 + logJ


# =============================================================================
def select_embed(shape, src, dim, index):
    """A function introduced as a replacement of `torch.select_scatter` that
    does not support automatic differentiation for outputs with complex dtype.
    """
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    for ell in range(shape[-2]):
        out[..., ell, ell] = 1
    out[[slice(None)] * dim + [index]] = src
    return out


def matrix_product(tuple_, right_product=True):
    """Return the matrix product of the matrices in the input tuple."""
    if len(tuple_) == 0:
        return 1
    else:
        product = tuple_[0]

    for x in tuple_[1:]:
        if right_product:
            product = product @ x
        else:
            product = x @ product
    return product
