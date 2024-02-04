# Copyright (c) 2021-2023 Javad Komijani

"""This module has utilities to handle staples related to gauge links."""


import torch
from ..linalg import special_svd

mul = torch.matmul


# =============================================================================
class TemplateStaplesHandle:

    def __init__(self, onesided=True):
        self.onesided = onesided

    def staple(self, link, *, staples_object):
        """
        Return `slink` (stapled link) defined as `link` multiplied by SU(n)
        matrices that are obtained by performing SVD on the sum of the
        corresponding `staples.`
        """
        # link = link @ staples_object.factorized_staple

        svd_ = special_svd(staples_object.data)
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

        # link = link @ staples_object.factorized_staple.adjoint()

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
class StaplesObject:

    def __init__(self, data, factorized_staple=None, extra=None):
        self.data = data
        self.extra = extra
        self.factorized_staple = factorized_staple


class WilsonStaplesHandle(TemplateStaplesHandle):

    vector_axis = 0

    def makesure_correct_vector_axis(self, vector_axis):
        assert self.vector_axis == vector_axis, "vector axis?"

    @classmethod
    def calc_staples(cls, links, *, mu, nu_list, extra_coeffs_list=None):
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

        if extra_coeffs_list is None:

            data = sum(
               [cls.calc_planar_staples(links, mu=mu, nu=nu) for nu in nu_list]
               )

            extra = None
        else:
            len_ = 2 * len(nu_list)
            all_staples = [None] * len_
            for k, nu in enumerate(nu_list):
                all_staples[2*k] = cls.calc_planar_staples(
                    links, mu=mu, nu=nu, up_only=True
                    )
                all_staples[2*k + 1] = cls.calc_planar_staples(
                    links, mu=mu, nu=nu, down_only=True
                    )

            # factorized_staple = all_staples[0]
            # eye = torch.eye(links[0].shape[-1], device=links[0].device)
            # links might be a list or a tensor
            # data = eye + factorized_staple.adjoint() @ sum(all_staples[1:])

            factorized_staple = None
            data = sum(all_staples)

            # extra = torch.empty(*data.shape[:-2], 1 + len(extra_coeffs_list))
            extra = torch.zeros(*data.shape[:-2], 1 + 2 * len(nu_list))

            extra[..., 0] = torch.linalg.matrix_norm(data)

            for k, nu in enumerate(nu_list):
                long_loop = all_staples[2*k] @ all_staples[2*k+1].adjoint()
                extra[..., 2*k + 1: 2*k + 3] = \
                        torch.view_as_real(calc_reduced_trace(long_loop))

            # for k, coeffs in enumerate(extra_coeffs_list):
            #    extra[..., 1+k] = torch.linalg.matrix_norm(
            #            sum([all_staples[j] * coeffs[j] for j in range(len_)])
            #            )

        return StaplesObject(data, factorized_staple, extra)

    @classmethod
    def calc_planar_staples(
            cls, links, *, mu, nu, up_only=False, down_only=False
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

        if cls.vector_axis == 0:
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
            return cls.staple1_rule(a, b, c)

        elif down_only:
            e = torch.roll(u, +1, dims=1 + nu)
            f = torch.roll(c, +1, dims=1 + nu)
            d = torch.roll(f, -1, dims=1 + mu)
            return cls.staple2_rule(d, e, f)

        else:
            a = torch.roll(c, -1, dims=1 + mu)
            b = torch.roll(u, -1, dims=1 + nu)
            e = torch.roll(u, +1, dims=1 + nu)
            f = torch.roll(c, +1, dims=1 + nu)
            d = torch.roll(f, -1, dims=1 + mu)
            return cls.staple1_rule(a, b, c) + cls.staple2_rule(d, e, f)

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
def calc_trace(x):
    return torch.sum(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)


def calc_reduced_trace(x):  # reduced trace = 1/n trace()
    return torch.mean(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)
