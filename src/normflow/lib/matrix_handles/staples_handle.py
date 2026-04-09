# Copyright (c) 2021-2026 Javad Komijani

"""This module has utilities to handle staples related to gauge links."""

import torch
from lattice_ml.functions import naive_project_onto_su3
from ..linalg import special_svd

mul = torch.matmul


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

    def staple(self, link, *, staples_object):
        """
        Staple a link by attaching the staple contribution to it.

        This operation combines the input `link` with a matrix derived from the
        surrounding staples (via SVD), producing a "stapled link" (`slink`).

        Parameters
        ----------
        link : tensor-like
            Gauge link matrix (typically SU(n)).
        staples_object : StaplesObject
            Object containing the staples sum and associated metadata.

        Returns
        -------
        slink : tensor-like
            Stapled link obtained by combining link with the projected staples.

        Notes
        -----
        - The SVD of the staples is cached in `staples_object.svd_`.
        - For `onesided=True`:  slink = link @ (SVD projection)
        - For `onesided=False`: slink = Vh @ link @ U
        """
        svd_ = special_svd(staples_object.data())
        staples_object.svd_ = svd_

        if self.onesided:
            slink = link @ svd_.sUVh  # slink stands for stapled link
        else:
            slink = svd_.Vh @ link @ svd_.sU

        return slink

    def unstaple(self, slink, *, staples_object):
        """
        Recover the original link from a stapled link.

        This is the inverse operation of `staple`, using the cached SVD stored
        in `staples_object`.

        Parameters
        ----------
        slink : tensor-like
            Stapled link.
        staples_object : StaplesObject
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
        svd_ = staples_object.svd_

        if self.onesided:
            link = slink @ svd_.sUVh.adjoint()
        else:
            link = svd_.Vh.adjoint() @ slink @ svd_.sU.adjoint()

        return link

    def push2link(self, link, *, slink_rotation, staples_object):
        """
        Apply a rotation in stapled-link space back to the original link.

        Parameters
        ----------
        link : tensor-like
            Original gauge link.
        slink_rotation : tensor-like
            Rotation (update) computed in stapled-link space.
        staples_object : StaplesObject
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
            svd_ = staples_object.svd_
            slink_rotation = svd_.Vh.adjoint() @ slink_rotation @ svd_.Vh

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
        self.svd_ = special_svd(staples)
        self.suvh = svd_.sU @ svd_.Vh

    def staple(self, link):
        """
        Apply the fixed staple transformation to a link.
        """
        slink = link @ self.suvh  # slink stands for stapled link
        return slink, self.svd_

    def unstaple(self, slink):
        """
        Invert the staple transformation.
        """
        return slink @ self.suvh.adjoint()


# =============================================================================
class WilsonStaplesHandle(TemplateStaplesHandle):
    """
    Staples handler for the Wilson gauge action.

    Provides methods to compute staples from lattice gauge links by summing
    contributions over mu-nu plaquettes.
    """

    link_axis = 1

    def calc_staples_sum(self, *args, **kwargs):
        """
        Convenience method returning only the staples sum.

        Returns
        -------
        tensor-like
            Sum of staples over all specified planes.
        """
        return self.calc_staples(*args, **kwargs).staples_sum

    def makesure_correct_link_axis(self, link_axis):
        """
        Validate that the provided link axis matches the expected one.
        """
        assert self.link_axis == link_axis, "link axis?"

    def calc_staples(
        self, links, *, mu, nu_list,
        staples_coeff=None, mixed_staples_coeff=None
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
        StaplesObject
            Object containing the computed staples and metadata.
        """

        if staples_coeff is None:
            staples_sum = sum(
              [self.calc_planar_staples(links, mu=mu, nu=nu) for nu in nu_list]
            )
            kwargs = dict(mu=mu, mixed_staples_coeff=mixed_staples_coeff)
            return StaplesObject(staples_sum, **kwargs)

        else:
            all_staples = [None] * (2 * len(nu_list))
            for k, nu in enumerate(nu_list):
                all_staples[2*k] = self.calc_planar_staples(
                    links, mu=mu, nu=nu, up_only=True
                    )
                all_staples[2*k + 1] = self.calc_planar_staples(
                    links, mu=mu, nu=nu, down_only=True
                    )

            range_ = range(len(all_staples))
            staples = sum([all_staples[j] * staples_coeff[j] for j in range_])
            return StaplesObject(staples)

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
        else:
            # then assume it is 1
            x_mu = links[:, mu]
            x_nu = links[:, nu]

        u = x_mu  # U in the above graph
        c = x_nu

        if up_only:
            a = torch.roll(c, -1, dims=1 + mu)
            b = torch.roll(u, -1, dims=1 + nu)
            return self.staple1_rule(a, b, c)

        elif down_only:
            e = torch.roll(u, +1, dims=1 + nu)
            f = torch.roll(c, +1, dims=1 + nu)
            d = torch.roll(f, -1, dims=1 + mu)
            return self.staple2_rule(d, e, f)

        else:
            a = torch.roll(c, -1, dims=1 + mu)
            b = torch.roll(u, -1, dims=1 + nu)
            e = torch.roll(u, +1, dims=1 + nu)
            f = torch.roll(c, +1, dims=1 + nu)
            d = torch.roll(f, -1, dims=1 + mu)
            return self.staple1_rule(a, b, c) + self.staple2_rule(d, e, f)

    @staticmethod
    def staple1_rule(a, b, c):
        r"""return :math:`a  b^\dagger  c^\dagger`."""
        #   --b--
        #  c|   |a
        #   @ U @
        return mul(a, mul(c, b).adjoint())

    @staticmethod
    def staple2_rule(d, e, f):
        r"""return :math:`d^\dagger  e^\dagger  f`."""
        #   @ U @
        #  f|   |d
        #   --e--
        return mul(mul(e, d).adjoint(), f)


# =============================================================================
class U1WilsonStaplesHandle(WilsonStaplesHandle):
    """Properties and methods are chosen to be consistent with SU(n)."""

    def __init__(self):
        pass  # not ready

    def staple(self):
        pass  # not ready

    def unstaple(self):
        pass  # not ready

    @staticmethod
    def staple1_rule(a, b, c):
        return a * torch.conj(b * c)

    @staticmethod
    def staple2_rule(d, e, f):
        return torch.conj(e * d) * f


# =============================================================================
class StaplesObject:
    """
    Container for staples and related derived quantities.

    Provides access to:
    - raw staples sum
    - optional mixed (higher-order) staples
    - cached SVD for transformations
    """
    svd_ = None

    def __init__(self, staples_sum, mu=None, mixed_staples_coeff=None):
        self.staples_sum = staples_sum
        self.mixed_staples_coeff = mixed_staples_coeff
        self.mu = mu

    def data(self):
        """
        Return the effective staples used for SVD.

        Includes mixed staples if coefficients are provided.

        Returns
        -------
        tensor-like
            Effective staples matrix.
        """
        coeff = self.mixed_staples_coeff
        if coeff is None:
            return self.staples_sum
        else:
            mixed = self.mixedstaples()
            return self.staples_sum + coeff['1'] * mixed['1'] \
                    + coeff['2'] * mixed['2'] + coeff['3'] * mixed['3']

    def mixedstaples(self):
        """
        Compute higher-order (mixed) staples contributions.

        Returns
        -------
        dict[str, tensor-like]
            Dictionary with keys '1', '2', '3' corresponding to different
            loop structures.
        """
        gamma = self.staples_sum
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
        gamma = self.staples_sum
        loop_left = torch.roll(gamma @ link, 1, dims=1 + self.mu)
        loop_right = torch.roll(link @ gamma, -1, dims=1 + self.mu)
        out_dict = {
            '2': gamma @ loop_left + loop_right @ gamma,
            '2d': gamma @ loop_left.adjoint() + loop_right.adjoint() @ gamma,
            '3': loop_right @ gamma @ loop_left
            }
        return out_dict

    @property
    def singv(self):
        """Return singular values"""
        try:
            svd_ = self.svd_
        except:
            raise NameError()

        if svd_.S.shape[-1] == 2:
            singv = svd_.S[..., :1]
        else:
            singv = torch.cat([svd_.S, svd_.rdet_angle.unsqueeze(-1)], -1)
        return singv

    def get_dual_param(self, eigvecs):
        """Return singular values"""
        try:
            svd_ = self.svd_
        except:
            raise NameError()

        if svd_.S.shape[-1] == 2:
            dual = svd_.S[..., :1]
        else:
            sigma = torch.linalg.diagonal(
                eigvecs.adjoint() @ svd_.Sigma @ eigvecs
                ).real
            alpha = svd_.rdet_angle.unsqueeze(-1)
            dual = torch.cat([sigma, torch.cos(alpha), torch.sin(alpha)], -1)
        return dual


# =============================================================================
def calc_trace(x):
    return torch.sum(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)


def calc_reduced_trace(x):  # reduced trace = 1/n trace()
    return torch.mean(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)
