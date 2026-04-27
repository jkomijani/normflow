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


__all__ = [
    "GaugeModuleList_",
    "GaugeSLinkModule_",
    "SpectralStateTransform_"
]


# =============================================================================
class GaugeModuleList_(ModuleList_):
    """
    Container for a sequence of `GaugeSLinkModule_` transformations.

    Applies gauge updates sequentially along a chosen link axis.

    Attributes
    ----------
    unbind_link_axis : bool
        If True, the input is split along `link_axis` into a list of tensors
        before applying the modules, and stacked back afterward. This lets each
        `GaugeSLinkModule_` operate on a single link direction (internally
        assuming link_axis = 0).
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
        list_of_gauge_modules_ : list[GaugeSLinkModule_]
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
        if not self.unbind_link_axis:
            return super().forward(x.clone(), log0)

        x = list(torch.unbind(x, self.link_axis))
        x, log0 = super().forward(x, log0)
        return torch.stack(x, dim=self.link_axis), log0

    def reverse(self, x, log0=0):
        """
        Apply inverse transformations in reverse order.

        Mirrors `forward`, including optional splitting along `link_axis`.
        Returns updated `x` and accumulated log-Jacobian.
        """
        if not self.unbind_link_axis:
            return super().reverse(x.clone(), log0)

        x = list(torch.unbind(x, self.link_axis))
        x, log0 = super().reverse(x, log0)
        return torch.stack(x, dim=self.link_axis), log0

    def hack(self, x, log0=0):
        """Similar to the forward method, except that returns the output of
        middle blocks too; useful for examining effects of each block.
        """
        stack = [(x, log0)]

        if not self.unbind_link_axis:
            return None

        for net_ in self:
            x = list(torch.unbind(x, self.link_axis))
            x, log0 = net_(x, log0)
            x = torch.stack(x, dim=self.link_axis)
            stack.append([x, log0])
        return stack


# =============================================================================
class GaugeSLinkModule_(Module_):
    """
    Gauge-equivariant link update via an invertible spectral transformation.

    For links in direction `mu`, the module:
        1. Builds a "stapled link" (slink) using local staple information.
        2. Applies an invertible transformation in spectral space.
        3. Converts the result into a rotation and updates the original links.

    The spectral transform operates on eigen-angles and eigenvectors and may
    include learned components (param/eigang/eigvec networks).

    Parameters
    ----------
    mu : int
        Direction of links to update.

    nu_list : list[int]
        Directions defining planes (with mu) used to construct staples.

    staples_handle : object
        Provides staple computation, slink construction, and push-back.

    slink_transform_ : Module_
        Pipline for transforming the slink.

    staples_kwargs : dict or None
        Extra arguments forwarded to staple computation.

    Notes
    -----
    The transformation is invertible if all constituent networks are invertible
    and use compatible masking.
    """

    unbounded_link_axis = True  # "link_axis" of inputs is supposed to be 0
    sites_before_link = True  # It is irrelavant if unbounded_link_axis is True

    def __init__(
        self,
        mu,
        nu_list,
        staples_handle,
        slink_transform_,
        staples_kwargs=None
    ):
        super().__init__()
        self.mu = mu
        self.nu_list = nu_list

        self.slink_transform_ = slink_transform_

        self.staples_handle = staples_handle
        self.staples_kwargs = staples_kwargs or {}

        # Resolve and set link axis convention
        self.link_axis = self._resolve_link_axis()
        self.staples_handle.link_axis = self.link_axis

    def forward(self, x, log0=0):
        """Apply forward link update."""
        return self._update_links(x, log0, reverse=False)

    def reverse(self, x, log0=0):
        """Apply inverse link update."""
        return self._update_links(x, log0, reverse=True)

    def _update_links(self, x, log0, reverse):
        """
        Apply a forward or inverse update to links in direction `mu`.

        Pipeline:
            staples → slink → spectral transform → update slink → update link

        Args:
            x (tensor-like): Input gauge field.
            log0 (tensor-link): Initial log-Jacobian accumulator.
            reverse (bool): If True, applies the inverse transformation.

        Returns:
            x (tensor-like): Updated gauge field
            logj (tensor-like): Accumulated log-Jacobian.
        """
        # Compute staple context (data + cached helpers)
        staples_ctx = self._compute_staples(x)

        # Extract links in direction mu
        x_mu = self._get_x_mu(x)

        # Build stapled link (link + projected staple contribution)
        slink = self.staples_handle.staple(x_mu, staples_ctx)

        # Apply invertible spectral transform
        transform = self.slink_transform_
        if reverse:
            slink, logj = transform.reverse(slink, log0, staples_ctx)
        else:
            slink, logj = transform.forward(slink, log0, staples_ctx)

        # Convert slink update into a rotation and push back to link
        x_mu = self.staples_handle.unstaple(slink, staples_ctx)

        # Write updated links back into full field
        x = self._set_x_mu(x, x_mu)

        return x, logj

    def _compute_staples(self, x):
        """Return staple context (data and helpers) for link update."""
        return self.staples_handle.compute_directional_staples_ctx(
            x, mu=self.mu, nu_list=self.nu_list, **self.staples_kwargs
        )

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
class SpectralStateTransform_(Module_):
    """
    Applies a sequence of invertible transformations to the spectral state.

    The spectral transform operates on eigen-angles and eigenvectors and may
    include learned components (param/eigang/eigvec networks).

    This module is a *pure composition engine*:
        _ The decomposition/recomposition is handled by `matrix_handle`.

    Parameters
    ----------
    matrix_handle : object
        Handles spectral decomposition and reconstruction of matrices.

    param_net_ : Module_ or None
        Network acting in parameter space (via eigang ↔ param mapping).

    dual_param_net_ : Module_ or None
        Optional secondary network conditioned on dual parameters.

    ops : torch.nn.Module or None
        A list of additional transformations that implement:
            forward(state, staples_ctx) -> state
            reverse(state, staples_ctx) -> state

    Notes:
       The transform is invertible if and only if all ops are invertible.
       If masks are used, they must be comptaible.
    """

    def __init__(
        self,
        matrix_handle,
        param_net_,
        dual_param_net_=None,
        extra_ops=None
    ):
        super().__init__()

        self.matrix_handle = matrix_handle

        # Build spectral transformation pipeline
        ops = []

        if param_net_ is not None or dual_param_net_ is not None:
            op = ParamTransformOp(matrix_handle, param_net_, dual_param_net_)
            ops.append(op)

        if extra_ops is not None:
            ops = ops + extra_ops

        self.ops = torch.nn.ModuleList(ops)

    def forward(self, x, log0=0, staples_ctx=None):
        """
        Apply forward update on input matrix x.

        Args:
            x (tensor-like): Input field.
            log0 (tensor-link): Initial log-Jacobian accumulator.
            staples_ctx (object): Staple context (cached decompositions).

        Returns:
            x (tensor-like): Updated matrix.
            logj (tensor-like): Accumulated log-Jacobian.
        """

        # Spectral representation
        state = self.matrix_handle.matrix_to_spectral_state(x)

        # Sequentially apply all forward transforms
        for op in self.ops:
            state = op.forward(state, staples_ctx)

        # Reconstruct transformed x
        x, logj = self.matrix_handle.spectral_state_to_matrix_(state)

        return x, log0 + logj

    def reverse(self, x, log0=0, staples_ctx=None):
        """
        Apply invers update on input matrix x.

        Args:
            x (tensor-like): Input field.
            log0 (tensor-link): Initial log-Jacobian accumulator.
            staples_ctx (object): Staple context (cached decompositions).

        Returns:
            x (tensor-like): Updated matrix.
            logj (tensor-like): Accumulated log-Jacobian.
        """

        # Spectral representation
        state = self.matrix_handle.matrix_to_spectral_state(x)

        # Reverse order is required for correct inverse composition
        for op in reversed(self.ops):
            state = op.reverse(state, staples_ctx)

        # Reconstruct transformed x
        x, logj = self.matrix_handle.spectral_state_to_matrix_(state)

        return x, log0 + logj


# =============================================================================
class ParamTransformOp(torch.nn.Module):
    """
    Handles transformation between eigen-angle space and parameter space.

    This stage:
        1. maps eigen-angles → parameters
        2. applies learned parameter transformation
        3. optionally applies dual conditioning from staples context
        4. maps back to eigen-angle space
    """

    def __init__(self, matrix_handle, param_net_, dual_param_net_=None):
        super().__init__()
        self.matrix_handle = matrix_handle
        self.param_net_ = param_net_
        self.dual_param_net_ = dual_param_net_

    def forward(self, state, staples_ctx):
        """
        Apply forward transformation.

        Args:
            state (SpectralState): Contains eigangs, eigvecs, and logj.
            staples_ctx (object): Staple context (cached decompositions).

        Returns:
            SpectralState: Updated spectral state after transformation.
        """
        if self.param_net_ is None and self.dual_param_net_ is None:
            return state

        eigangs = state.eigangs
        eigvecs = state.eigvecs

        param, logj_ang2par = self.matrix_handle.eigang2param_(eigangs)

        if self.param_net_ is not None:
            param, logj_par2par = self.param_net_(param)
        else:
            logj_par2par = 0

        if self.dual_param_net_ is not None:
            param, logj_par2par = self.dual_param_net_(
                param, staples_ctx.get_dual_param(eigvecs), log0=logj_par2par
            )

        eigangs, logj_par2ang = self.matrix_handle.param2eigang_(param)

        state.eigangs = eigangs
        state.logj += (logj_ang2par + logj_par2par + logj_par2ang)
        return state

    def reverse(self, state, staples_ctx):
        """
        Apply inverse transformation.

        Args:
            state (SpectralState): Contains eigangs, eigvecs, and logj.
            staples_ctx (object): Staple context (cached decompositions).

        Returns:
            SpectralState: Updated spectral state after inverse transformation.
        """

        if self.param_net_ is None and self.dual_param_net_ is None:
            return state

        eigangs = state.eigangs
        eigvecs = state.eigvecs

        param, logj_ang2par = self.matrix_handle.eigang2param_(eigangs)

        if self.dual_param_net_ is not None:
            param, logj_par2par = self.dual_param_net_.reverse(
                param, staples_ctx.get_dual_param(eigvecs)
            )
        else:
            logj_par2par = 0

        if self.param_net_ is not None:
            param, logj_par2par = self.param_net_.reverse(
                param, log0=logj_par2par
            )

        eigangs, logj_par2ang = self.matrix_handle.param2eigang_(param)

        state.eigangs = eigangs
        state.logj += (logj_ang2par + logj_par2par + logj_par2ang)
        return state


# =============================================================================
class _PolyakovGaugeModule_(MatrixModule_):

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
        """Forward pass"""

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
        """Reverse pass"""

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

    product = tuple_[0]

    for x in tuple_[1:]:
        if right_product:
            product = product @ x
        else:
            product = x @ product
    return product
