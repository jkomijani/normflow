# Copyright (c) 2021-2024 Javad Komijani

"""This module has utilities to handle staples related to gauge links."""


import torch
from ..linalg import special_svd

mul = torch.matmul


# =============================================================================
class TemplateStaplesHandle:

    onesided = True

    def staple(self, link, *, staples_object):
        """
        Return `slink` (stapled link) defined as `link` multiplied by SU(n)
        matrices that are obtained by performing SVD on the sum of the
        corresponding `staples.`
        """
        svd_ = special_svd(staples_object.data())
        staples_object.svd_ = svd_

        if self.onesided:
            slink = link @ svd_.sUVh  # slink stands for stapled link
        else:
            slink = svd_.Vh @ link @ svd_.sU

        return slink

    def unstaple(self, slink, *, staples_object):
        """Invert the `staple` method.

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

        if not self.onesided:
            svd_ = staples_object.svd_
            slink_rotation = svd_.Vh.adjoint() @ slink_rotation @ svd_.Vh

        return slink_rotation @ link


# =============================================================================
class FixedStaplesHandle:

    def __init__(self, staples):
        self.svd_ = special_svd(staples)
        self.suvh = svd_.sU @ svd_.Vh

    def staple(self, link):
        slink = link @ svd_.suvh  # slink stands for stapled link
        return slink, self.svd_

    def unstaple(self, slink):
        return slink @ self.suvh.adjoint()


# =============================================================================
class WilsonStaplesHandle(TemplateStaplesHandle):

    vector_axis = 1

    def calc_staples_sum(self, *args, **kwargs):
        return self.calc_staples(*args, **kwargs).staples_sum

    def makesure_correct_vector_axis(self, vector_axis):
        assert self.vector_axis == vector_axis, "vector axis?"

    def calc_staples(self, links, *, mu, nu_list, staples_coeff=None,
            mixed_staples_coeff=None):
        """Calculate the staples (from the Wilson gauge action) corresponding
        to the `links` that are in `mu` direction and summed over mu-nu planes
        with nu in `nu_list`.

        Stables of the Wilson gauge action in any plane are shown in the
        following cartoon:

            >>>     --b--
            >>>    c|   |a
            >>>     @ U @    +   @ U @
            >>>                 f|   |d
            >>>                  --e--

        where `@ U @` shows the central link for which the staples are going to
        be calculated.

        Parameters
        ----------
        links : tensor
            Tensor of gauge links.
        mu : int
            Direction of the links with them the staples are associated.
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
        """Similar to calc_staples, except that the staples are calculated on
        mu-nu plane.
        """
        # In the plane specified with mu and nu, calculate the staples
        # $a 1/b 1/c$ and $1/d 1/e f$
        #
        #   --b--
        #  c|   |a
        #   @ U @    +    @ U @
        #                f|   |d
        #                 --e--

        if self.vector_axis == 0:
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
        """return :math:`a  b^\dagger  c^\dagger`."""
        #   --b--
        #  c|   |a
        #   @ U @
        return mul(a, mul(c, b).adjoint())

    @staticmethod
    def staple2_rule(d, e, f):
        """return :math:`d^\dagger  e^\dagger  f`."""
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

    svd_ = None

    def __init__(self, staples_sum, mu=None, mixed_staples_coeff=None):
        self.staples_sum = staples_sum
        self.mixed_staples_coeff = mixed_staples_coeff
        self.mu = mu

    def data(self):
        coeff = self.mixed_staples_coeff
        if coeff is None:
            return self.staples_sum
        else:
            mixed = self.mixedstaples()
            return self.staples_sum + coeff['1'] * mixed['1'] \
                    + coeff['2'] * mixed['2'] + coeff['3'] * mixed['3']

    def mixedstaples(self):
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
