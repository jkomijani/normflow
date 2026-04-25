# Copyright (c) 2021-2026 Javad Komijani

"""This module has utilities to handle staples related to gauge links."""

# pylint: disable=invalid-name


from dataclasses import dataclass, field
import torch
from lattice_ml.functions import naive_project_onto_su3
from ..linalg import compute_svd

matmul = torch.matmul


__all__ = ["WilsonStaplesHandle"]


# =============================================================================
class TemplateStaplesHandle:
    """
    Base class implementing the transformation between gauge links and their
    "stapled" versions using SVD-projected staples.

    This class defines a reversible mapping:
        link <-> stapled link (slink)

    The mapping depends on the SVD of the associated staples and can be either
    one-sided or two-sided depending on `onesided`, which defaults to True.
    """

    onesided = True

    def staple(self, link, staples_ctx):
        """
        Staple a link by attaching the staple contribution to it.

        This operation combines the input `link` with a matrix derived from the
        surrounding staples (via SVD), producing a "stapled link" (`slink`).

        Parameters
        ----------
        link : tensor-like
            Gauge link matrix (typically SU(n)).
        staples_ctx : StaplesContext
            Object containing the staples sum and associated metadata.

        Returns
        -------
        slink : tensor-like
            Stapled link obtained by combining link with the projected staples.

        Notes
        -----
        - The SVD of the staples is cached in `staples_ctx.svd_result`.
        - For `onesided=True`:  slink = link @ (SVD projection)
        - For `onesided=False`: slink = Vh @ link @ U
        """
        svd_result = staples_ctx.svd_result

        if self.onesided:
            slink = link @ svd_result.special_unitary_factor
        else:
            d = svd_result.diagonal_phase_factor  # diagonals of D
            UDh = svd_result.U * d.conj().unsqueeze(-2)
            slink = svd_result.Vh @ link @ UDh

        return slink

    def unstaple(self, slink, staples_ctx):
        """
        Recover the original link from a stapled link.

        This is the inverse operation of `staple`, using the cached SVD stored
        in `staples_ctx`.

        Parameters
        ----------
        slink : tensor-like
            Stapled link.
        staples_ctx : StaplesContext
            Object containing the cached SVD.

        Returns
        -------
        link : tensor-like
            Reconstructed original gauge link.

        Notes
        -----
        For pushing the changes in `slink` to corresponding `link` use the
        `push2link` method.
        """
        svd_result = staples_ctx.svd_result

        if self.onesided:
            link = slink @ svd_result.special_unitary_factor.adjoint()
        else:
            d = svd_result.diagonal_phase_factor  # diagonals of D
            UDh = svd_result.U * d.conj().unsqueeze(-2)
            link = svd_result.Vh.adjoint() @ slink @ UDh.adjoint()

        if link.shape[-1] == 3:
            link = naive_project_onto_su3(link)  # correct numerical deviations

        return link

    def push2link(self, link, slink_rotation, staples_ctx):
        """
        Apply a rotation in stapled-link space back to the original link.

        Parameters
        ----------
        link : tensor-like
            Original gauge link.
        slink_rotation : tensor-like
            Rotation (update) computed in stapled-link space.
        staples_ctx : StaplesContext
            Object containing SVD data for consistency.

        Returns
        -------
        updated_link : tensor-like
            Updated link after applying the rotation.

        Notes
        -----
        - For two-sided transformations, the rotation is mapped back using SVD.
        - For SU(3), a projection is applied to maintain group structure.
        """
        if not self.onesided:
            Vh = staples_ctx.svd_result.Vh
            slink_rotation = Vh.adjoint() @ slink_rotation @ Vh

        if link.shape[-1] == 3:
            # Projection to SU(3) to correct numerical deviations
            return naive_project_onto_su3(slink_rotation @ link)

        return slink_rotation @ link


# =============================================================================
class FixedStaplesHandle:
    """
    Variant of staples handler with precomputed (fixed) staples.

    Unlike `TemplateStaplesHandle`, the SVD is computed once during
    initialization and reused for all operations.
    """

    def __init__(self, staples):
        """
        Parameters
        ----------
        staples : tensor-like
            Precomputed staples sum used to define the transformation.
        """
        self.svd_result = compute_svd(staples)

    def staple(self, link):
        """
        Apply the fixed staple transformation to a link.
        """
        slink = link @ self.svd_result.special_unitary_factor
        return slink

    def unstaple(self, slink):
        """
        Invert the staple transformation.
        """
        return slink @ self.svd_result.special_unitary_factor.adjoint()


# =============================================================================
class WilsonStaplesHandle(TemplateStaplesHandle):
    """
    Staples handler for the Wilson gauge action.

    Provides methods to compute staples from lattice gauge links by summing
    contributions over mu-nu plaquettes.
    """

    link_axis = -3  # indicates the site axes are before the link axis

    def compute_directional_staples_ctx(
        self, links, mu, nu_list, staples_coeff=None, mixed_staples_coeff=None
    ):
        """
        Compute the sum of Wilson staples for links in direction `mu`.

        Staples are constructed from plaquettes in all `mu-nu` planes as shown
        in the following cartoon:

            >>>     --b--
            >>>    c|   |a
            >>>     @ U @    +   @ U @
            >>>                 f|   |d
            >>>                  --e--

        where `@ U @` shows the central link for which the staples are going to
        be calculated.

        Parameters
        ----------
        links : tensor-like
            Lattice gauge field.
        mu : int
            Direction of the central link.
        nu_list : list[int]
            Directions defining planes with `mu`.
        staples_coeff : list[float], optional
            Weights for individual staple contributions.
        mixed_staples_coeff : dict, optional
            Coefficients for higher-order (mixed) staples.

        Returns
        -------
        StaplesContext
            Object containing the computed staples and metadata.
        """

        if staples_coeff is None:
            staples = sum(
                self.calc_planar_staples(links, mu=mu, nu=nu) for nu in nu_list
            )

        else:
            staples = [None] * (2 * len(nu_list))
            for k, nu in enumerate(nu_list):
                staples[2*k] = self.calc_planar_staples(
                    links, mu=mu, nu=nu, up_only=True
                )
                staples[2*k + 1] = self.calc_planar_staples(
                    links, mu=mu, nu=nu, down_only=True
                )

        staples_ctx = StaplesContext(
            staples, mu, staples_coeff, mixed_staples_coeff
        )
        return staples_ctx

    def calc_planar_staples(
        self, links, *, mu, nu, up_only=False, down_only=False
    ):
        """
        Compute staples in a single mu-nu plane.

        Parameters
        ----------
        links : tensor-like
            Lattice gauge field.
        mu, nu : int
            Directions defining the plane.
        up_only : bool, optional
            If True, compute only forward-oriented staple.
        down_only : bool, optional
            If True, compute only backward-oriented staple.

        Returns
        -------
        tensor-like
            Staples contribution in the specified plane.
        """
        # In the plane specified with mu and nu, calculate the staples
        # $a 1/b 1/c$ and $1/d 1/e f$
        #
        #   --b--
        #  c|   |a
        #   @ U @    +    @ U @
        #                f|   |d
        #                 --e--

        if self.link_axis == 0:
            x_mu = links[mu]
            x_nu = links[nu]
        elif self.link_axis == 1:
            x_mu = links[:, mu]
            x_nu = links[:, nu]
        else:
            # then assume it is -3
            x_mu = links[..., mu, :, :]
            x_nu = links[..., nu, :, :]

        u = x_mu  # U in the above graph
        c = x_nu

        if up_only:
            a = torch.roll(c, -1, dims=1 + mu)
            b = torch.roll(u, -1, dims=1 + nu)
            staple = self.staple1_rule(a, b, c)

        elif down_only:
            e = torch.roll(u, +1, dims=1 + nu)
            f = torch.roll(c, +1, dims=1 + nu)
            d = torch.roll(f, -1, dims=1 + mu)
            staple = self.staple2_rule(d, e, f)

        else:
            a = torch.roll(c, -1, dims=1 + mu)
            b = torch.roll(u, -1, dims=1 + nu)
            e = torch.roll(u, +1, dims=1 + nu)
            f = torch.roll(c, +1, dims=1 + nu)
            d = torch.roll(f, -1, dims=1 + mu)
            staple = self.staple1_rule(a, b, c) + self.staple2_rule(d, e, f)

        return staple

    @staticmethod
    def staple1_rule(a, b, c):
        r"""return :math:`a  b^\dagger  c^\dagger`."""
        #   --b--
        #  c|   |a
        #   @ U @
        return matmul(a, matmul(c, b).adjoint())

    @staticmethod
    def staple2_rule(d, e, f):
        r"""return :math:`d^\dagger  e^\dagger  f`."""
        #   @ U @
        #  f|   |d
        #   --e--
        return matmul(matmul(e, d).adjoint(), f)


# =============================================================================
class U1WilsonStaplesHandle(WilsonStaplesHandle):
    """Note Ready."""


# =============================================================================
@dataclass
class StaplesContext:
    """
    Container for staples and related derived quantities.

    Provides access to:
    - raw staples sum
    - optional mixed (higher-order) staples
    - cached SVD for transformations
    """

    _svd_result: torch.Tensor = field(default=None, init=False, repr=False)

    def __init__(
        self,
        staples,
        mu=None,
        staples_coeff=None,
        mixed_staples_coeff=None
    ):
        self.staples = staples
        self.staples_coeff = staples_coeff
        self.mixed_staples_coeff = mixed_staples_coeff
        self.mu = mu

    @property
    def svd_result(self):
        """Compute the singular value decomposition of the data."""
        if self._svd_result is None:
            self._svd_result = compute_svd(self._get_data())
        return self._svd_result

    def _get_data(self):
        """
        Return the effective staples used for SVD.

        Includes mixed staples if coefficients are provided.

        Returns
        -------
        tensor-like
            Effective staples matrix.
        """
        data = self.staples

        if self.staples_coeff is not None:
            data = sum(c * data[j] for j, c in enumerate(self.staples_coeff))

        if self.mixed_staples_coeff is not None:
            c = self.mixed_staples_coeff
            m = self.mixedstaples()
            data = data + c['1'] * m['1'] + c['2'] * m['2'] + c['3'] * m['3']

        return data

    def mixedstaples(self):
        """
        Compute higher-order (mixed) staples contributions.

        Returns
        -------
        dict[str, tensor-like]
            Dictionary with keys '1', '2', '3' corresponding to different
            loop structures.
        """
        gamma = self.staples
        gamma2 = gamma.adjoint() @ gamma
        loop_left = torch.roll(gamma @ gamma.adjoint(), 1, dims=1 + self.mu)
        loop_right = torch.roll(gamma2, -1, dims=1 + self.mu)
        out_dict = {
                '1': gamma @ gamma2,
                '2': loop_right @ gamma + gamma @ loop_left,
                '3': loop_right @ gamma @ loop_left
                }
        return out_dict

    def _mixedstaples(self, link):
        # Return G @ Gl @ L  +  R @ Gr @ G
        # where G, Gl and Gr are staples corresponding to U, L and R as
        # depicted in the following cartoon:
        #
        #  --Gl---G--       --G---Gr--
        #  |    !   |       |   !    |
        #  --L--@ U @   +   @ U @--R--
        #
        # and R @ Gr @ G @ Gl @ L from the following cartoon
        #
        #  --Gl---G---Gr--
        #  |    !   !    |
        #  --L--@ U @--R--
        #
        gamma = self.staples
        loop_left = torch.roll(gamma @ link, 1, dims=1 + self.mu)
        loop_right = torch.roll(link @ gamma, -1, dims=1 + self.mu)
        out_dict = {
            '2': gamma @ loop_left + loop_right @ gamma,
            '2d': gamma @ loop_left.adjoint() + loop_right.adjoint() @ gamma,
            '3': loop_right @ gamma @ loop_left
            }
        return out_dict

    def get_dual_param(self, eigvecs):
        """
        Return the dual parameters: coefficient of eigenvalues in the action.
        """
        svd_result = self.svd_result

        if svd_result.S.shape[-1] == 2:
            dual = svd_result.S[..., :1]
        else:
            sigma = svd_result.sigma_matrix_factor
            # `Σ = H + i λ I`, where H is Hermitian & λ is constant.
            dual = torch.view_as_real(
                torch.linalg.diagonal(eigvecs.adjoint() @ sigma @ eigvecs)
            ).reshape(*sigma.shape[:-2], -1)
        return dual


StaplesObject = StaplesContext  # for legacy
