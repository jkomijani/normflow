# Copyright (c) 2021-2022 Javad Komijani

"""This module includes splines utilites."""


from .._normflowcore import torch


class SplineTemplate:
    """Interpolate data with a piecewise function.

    Parameters
    ----------
    knots_x : tensor
        Array containing values of the independent variable.
    knots_y : tensor
        Array containing values of the dependent variable.
    knots_d : tensor
        Array containing derivatives w.r.t the independent variable.
    knots_axis : int, optional
        Axis along which coordinates of knots are assumed to be placed.

    extrapolate : bool, optional
        If True, extrapolates to out-of-bounds points, otherwise returns `NaN`.
        (The False option is not supported yet.)

    extrap_left : str, optional
        Relevant only when `extrapolate` is True.
        The extrapolation to the left points are based on the function obtained
        for the first segment, unless `extrap_left` is set to:
        `linear`,
        uses a linear function patched to the first segment;
        `periodic`,
        assumes a periodic boundary condition at the first knot;
        `anti-periodic` (or `anti`),
        assumes an anti-periodic boundary condition at the first knot;

    extrap_right : str, optional
        Similar to `extrap_left`, but for extrapolation to right.

    NOTE: Extrapolation based on first and last intervals can lead to poles.

    The number of knots must be at least equal 2.
    If the number of knots is 2 and `knots_d = None`, the spline will be
    basically a line.
    """
    def __init__(self, knots_x=None, knots_y=None, knots_d=None, knots_axis=-1,
            extrapolate=True, extrap_left=None, extrap_right=None
            ):

        if extrapolate is False:
            raise Exception("'extrapolate == False' is not supported yet!")

        if knots_x.shape != knots_y.shape:
            raise Exception("x and y must have the same shape.")

        if knots_d is None:
            knots_d = self.smooth_derivatives(knots_x, knots_y, knots_axis)

        if extrapolate:
            aug = AugmentKnots(knots_x, knots_y, knots_d, knots_axis)
            knots_x, knots_y, knots_d = aug(extrap_left, extrap_right)

        self.knots_x = knots_x
        self.knots_y = knots_y
        self.knots_d = knots_d
        self.knots_axis = knots_axis
        self.knots_len = knots_x.shape[knots_axis]
        self.segm_len = self.knots_len - 1
        self.shape = knots_x.shape
        self.extrapolate = extrapolate
        self.extrap_left = extrap_left
        self.extrap_right = extrap_right

    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def forward(self, x, grad=False, squeezed=False):
        """
        Parameters
        ----------
        x : tensor
            The input that is going to be transformed

        grad : bool, optional
            If True, the output will have the gradient of function in addition
            to the value of function at `x`. The default value is False.

        squeezed : bool, optional
            If False, the input `x` must have the same number of dimensions of
            `knots_x`. It is also required that for all axes except the
            `knots_axis` one the input `x` and `knots_x` have the same size.

            If squeezed is set to True, it is assumed that `x` does not have
            any dimensions corresponding to `knots_axis`; so, this method first
            unsqueezes `x`, performs the calculations, and finally squeezes the
            output back to the shape of `x'.

            The default value is False.
        """
        x = x.unsqueeze(self.knots_axis) if squeezed else x
        segments_ind = self.searchsorted(self.knots_x, x, self.knots_axis)
        kwargs = dict(squeezed=squeezed, grad=grad)
        func = self._calc_segment_func(segments_ind, **kwargs)
        return func(x)

    def backward(self, y, grad=False, squeezed=False):
        """Inverse of the forward method."""
        y = y.unsqueeze(self.knots_axis) if squeezed else y
        segments_ind = self.searchsorted(self.knots_y, y, self.knots_axis)
        kwargs = dict(squeezed=squeezed, grad=grad)
        inv_func = self._calc_segment_inv_func(segments_ind, **kwargs)
        return inv_func(y)

    @staticmethod
    def smooth_derivatives(knots_x, knots_y, knots_axis, bc_type='not-ones'):
        """For the internal knots, returns the average of the slopes of the
        the left and right segments. For the boundary knots, returns the slope
        of the nearby segment if `bc_type` is not 'ones', otherwise returns 1.
        """
        # Note that for n knots there are only n-1 segments
        knots_len = knots_x.shape[knots_axis]
        knot_ind = torch.tensor(range(knots_len))
        segm_ind = torch.tensor(range(knots_len - 1))
        select = lambda z, ind: torch.index_select(z, knots_axis, ind)
        diff_select = lambda z, ind: select(z, ind[1:]) - select(z, ind[:-1])
        sum_select = lambda z, ind: select(z, ind[1:]) + select(z, ind[:-1])

        # m is used to denote the slope of segments
        m = diff_select(knots_y, knot_ind) / diff_select(knots_x, knot_ind)
        m_avg = 0.5 * sum_select(m, segm_ind)

        if bc_type == 'ones':
            ones = torch.ones_like(select(m, segm_ind[0]))
            m_left, m_right = ones, ones
        else:
            m_left = select(m, segm_ind[0])
            m_right = select(m, segm_ind[-1])
        return torch.cat((m_left, m_avg, m_right), knots_axis)

    @staticmethod
    def searchsorted(x_sorted, x, axis):
        """The same `torch.searchsorted` except that the search can be done on
        any axis. (`torch.searchsorted` acts on the innermost dimension.)
        """
        if (axis == -1) or (axis == x.dim()-1):
            return torch.searchsorted(x_sorted, x)
        else:
            view_x_sorted = torch.movedim(x_sorted, axis, -1)
            view_x = torch.movedim(x, axis, -1)
            view_ind = torch.searchsorted(view_x_sorted, view_x)
            return torch.movedim(view_ind, -1, axis)


class Pade22Spline(SplineTemplate):
    """Rational quadratic, i.e. Pade [2, 2], spline data interpolator.

    Interpolate data with a piecewise rational quadratic which is
    monotonic and once (twice) continuously differentiable if `(x, y)` of knots
    are monotonically increasing and derivatives at knots are positive [1]_.
    To have a twice continously differentiable spline, the derivatices at knots
    can be free parameters only at two knots.
    """

    def _calc_segment_func(self, segm_ind, squeezed=False, grad=False):
        """Elements of segm_ind must be in range(self.segm_len + 2)."""

        axis = self.knots_axis
        segm_ind = torch.clamp(segm_ind, min=1, max=self.segm_len) - 1
        gather = lambda z, i: torch.gather(z, axis, segm_ind + i)

        x0 = gather(self.knots_x, 0)
        x1 = gather(self.knots_x, 1)
        y0 = gather(self.knots_y, 0)
        y1 = gather(self.knots_y, 1)
        d0 = gather(self.knots_d, 0)
        d1 = gather(self.knots_d, 1)
        m = (y1 - y0)/(x1 - x0)  # average slope of each segment

        squeezer = lambda y: y.squeeze(axis) if squeezed else y

        def g_0(theta):
            return y0 + (y1 - y0) * theta * (m * theta + d0 * (1 - theta)) \
                     / (m + (d1 + d0 - 2*m) * theta * (1 - theta))

        def g_1(theta):
            return m**2 * (d0 + 2 * (m - d0) * theta + (d1+d0-2*m) * theta**2) \
                     / (m + (d1 + d0 - 2*m) * theta * (1 - theta))**2

        def func(x):
            theta = (x - x0)/(x1 - x0)
            if grad:
                return squeezer(g_0(theta)), squeezer(g_1(theta))
            else:
                return squeezer(g_0(theta))

        return func

    def _calc_segment_inv_func(self, segm_ind, squeezed=False, grad=False):
        """Needs to be checked again to see if works correctly.
        (The sign of square root depends on details, which might not be
        specified correctly.)
        Elements of segm_ind must be in range(self.segm_len + 2)."""

        axis = self.knots_axis
        segm_ind = torch.clamp(segm_ind, min=1, max=self.segm_len) - 1
        gather = lambda z, i: torch.gather(z, axis, segm_ind + i)

        x0 = gather(self.knots_x, 0)
        x1 = gather(self.knots_x, 1)
        y0 = gather(self.knots_y, 0)
        y1 = gather(self.knots_y, 1)
        d0 = gather(self.knots_d, 0)
        d1 = gather(self.knots_d, 1)
        m = (y1 - y0)/(x1 - x0)  # average slope of each segment

        squeezer = lambda y: y.squeeze(axis) if squeezed else y

        def calc_theta(eta):
            """Calculate theta from eta, where
            eta   = (y - y0) / (y1 - y0) > 0
            theta = (x - x0) / (x1 - x0) > 0
            and they are related as
            (A eta - a) theta^2 + (B eta -b) theta + C eta = 0.
            """
            a2 = (2*m - d1 - d0) * eta + d0 - m
            a1 = -a2 - m
            a0 = m * eta
            delta = torch.sqrt(a1**2 - 4 * a0 * a2)
            theta = torch.empty(eta.shape)
            # conditions:
            # 1) 0 =< eta =< 1
            #    0 < m
            #    0 =< a0 
            # 2) a1**2 = (a2 + m)**2 = a2**2 + m**2 + 2 a2 m >= 4 a2 m
            #          >= 4 a2 m eta
            #    therefore ( a1**2 - 4 a2 a0) >= 0
            #    therefore delta is (positive) real
            # 3) if a2 >= 0: a1 < 0 and delta < |a1|
            # 4) if a2 < 0: delta > |a1|
            theta [a2 == 0] = (-a0/a1) [a2 == 0]
            theta [a2 != 0] = ((-a1 - delta)/(2 * a2)) [a2 != 0]
            return theta

        def g_1(theta):
            return m**2 * (d0 + 2 * (m - d0) * theta + (d1+d0-2*m) * theta**2) \
                     / (m + (d1 + d0 - 2*m) * theta * (1 - theta))**2

        def inv_func(y):
            eta = (y - y0)/(y1 - y0)
            theta = calc_theta(eta)
            x = x0 + (x1 - x0) * theta
            if grad:
                return squeezer(x), squeezer(1/g_1(theta))
            else:
                return squeezer(x)

        return inv_func


class Pade11Spline(SplineTemplate):
    """A simpler splines (compared to Pade22Spline) with continous derivatives.
    It can be useful if extrapolation is not required. Because there is not
    any control over derivatices (except at one knot), the extrapolation can
    be very wild due to very large derivatives at the end knots."""

    @staticmethod
    def smooth_derivatives(knots_x, knots_y, knots_axis, bc_type='natural'):

        knots_len = knots_x.shape[knots_axis]
        knot_ind = torch.tensor(range(knots_len))
        segm_ind = torch.tensor(range(knots_len - 1))
        select = lambda z, ind: torch.index_select(z, knots_axis, ind)
        diff_select = lambda z, ind: select(z, ind[1:]) - select(z, ind[:-1])

        m = diff_select(knots_y, knot_ind) / diff_select(knots_x, knot_ind)

        if bc_type == 'natural':
            n = 2 * ((knots_len-1)//2)  # segm_len = knots_len - 1
            d0 = torch.prod(
                select(m, knot_ind[1:n:2])/select(m, knot_ind[:n:2]), dim=knots_axis
                ).unsqueeze(knots_axis) * 0 + 1
        else:
            raise Exception("bc_type is not know")
        d_list = [d0]
        for k in range(knots_len - 1):
            d_list.append(select(m, knot_ind[k])**2 / d_list[-1])
        return torch.cat(tuple(d_list), knots_axis) 

    def _calc_segment_func(self, segm_ind, squeezed=False, grad=False):
        """Elements of segm_ind must be in range(self.segm_len + 2)."""

        axis = self.knots_axis
        segm_ind = torch.clamp(segm_ind, min=1, max=self.segm_len) - 1
        gather = lambda z, i: torch.gather(z, axis, segm_ind + i)

        x0 = gather(self.knots_x, 0)
        x1 = gather(self.knots_x, 1)
        y0 = gather(self.knots_y, 0)
        y1 = gather(self.knots_y, 1)
        d0 = gather(self.knots_d, 0)
        m = (y1 - y0)/(x1 - x0)  # average slope of each segment

        squeezer = lambda y: y.squeeze(axis) if squeezed else y

        def g_0(theta):
            return y0 + (y1 - y0) * d0 * theta / (m + (d0 - m) * theta)

        def g_1(theta):
            return m**2 * d0 / (m + (d0 - m) * theta)**2

        def func(x):
            theta = (x - x0)/(x1 - x0)
            if grad:
                return squeezer(g_0(theta)), squeezer(g_1(theta))
            else:
                return squeezer(g_0(theta))

        return func

    def _calc_segment_inv_func(self, segm_ind, squeezed=False, grad=False):
        """Elements of segm_ind must be in range(self.segm_len + 2)."""

        axis = self.knots_axis
        segm_ind = torch.clamp(segm_ind, min=1, max=self.segm_len) - 1
        gather = lambda z, i: torch.gather(z, axis, segm_ind + i)

        x0 = gather(self.knots_x, 0)
        x1 = gather(self.knots_x, 1)
        y0 = gather(self.knots_y, 0)
        y1 = gather(self.knots_y, 1)
        d0 = gather(self.knots_d, 0)
        m = (y1 - y0)/(x1 - x0)  # average slope of each segment

        squeezer = lambda y: y.squeeze(axis) if squeezed else y

        def g_1(theta):
            return m**2 * d0 / (m + (d0 - m) * theta)**2

        def inv_func(y):
            eta = (y - y0)/(y1 - y0)
            theta = -eta * m / (eta * (d0 - m) - d0)
            x = x0 + (x1 - x0) * theta
            if grad:
                return squeezer(x), squeezer(1/g_1(theta))
            else:
                return squeezer(x)

        return inv_func


RQSpline = Pade22Spline  # alias: Rational Quadratic Spline
RLSpline = Pade11Spline  # alias: Rational Linear Spline


class AugmentKnots:

    def __init__(self, knots_x, knots_y, knots_d, knots_axis):

        self.knots_x = knots_x
        self.knots_y = knots_y
        self.knots_d = knots_d
        self.knots_axis = knots_axis

    def __call__(self, left, right):
        self.perform_bc(left, right)
        return self.knots_x, self.knots_y, self.knots_d

    def perform_bc(self, left, right):

        if left is None and right is None:
            return
        elif (left == 'linear') or (right == 'linear'):
            self.takecare_linear(left, right)
            if left is None or right is None:
                return  # (left, right) are (None, 'linear') or ('linear', None)
        self.takecare_rest(left, right)

    def takecare_linear(self, left, right):
        x, y, d, axis = self.knots_x, self.knots_y, self.knots_d, self.knots_axis
        n = x.shape[axis]

        select_0 = lambda z: torch.index_select(z, axis, torch.tensor([0]))
        select_1 = lambda z: torch.index_select(z, axis, torch.tensor([n-1]))

        if left == "linear":
            x_fiducial_left = select_0(x) - 1
            y_fiducial_left = select_0(y) - select_0(d)
            d_fiducial_left = select_0(d)
        else:
            x_fiducial_left = None
            y_fiducial_left = None
            d_fiducial_left = None

        if right == "linear":
            x_fiducial_right = select_1(x) + 1
            y_fiducial_right = select_1(y) + select_1(d)
            d_fiducial_right = select_1(d)
        else:
            x_fiducial_right = None
            y_fiducial_right = None
            d_fiducial_right = None

        self.knots_x = self.cat([x_fiducial_left, x, x_fiducial_right], axis)
        self.knots_y = self.cat([y_fiducial_left, y, y_fiducial_right], axis)
        self.knots_d = self.cat([d_fiducial_left, d, d_fiducial_right], axis)

    def takecare_rest(self, left, right):
        x, y, d, axis = self.knots_x, self.knots_y, self.knots_d, self.knots_axis
        n = x.shape[axis]

        knot_ind = torch.tensor(range(n))
        select_0 = lambda z: torch.index_select(z, axis, knot_ind[:1])
        select_1 = lambda z: torch.index_select(z, axis, knot_ind[-1:])
        select = lambda z, ind: torch.index_select(z, axis, ind)
        selectflip = lambda z, ind: torch.flip(select(z, ind), [axis])

        if left in ["anti", "anti-periodic"]:
            x_fiducial_left = 2 * select_0(x) - selectflip(x, knot_ind[1:])
            y_fiducial_left = 2 * select_0(y) - selectflip(y, knot_ind[1:])
            d_fiducial_left = selectflip(d, knot_ind[1:])
        elif left == "periodic":
            # first check if derivative @ boundary is zero
            if not sum(torch.index_select(d, axis, knot_ind[:1]) == 0):
                raise Exception("Oops: derivative at periodic bc must be zero.")
            x_fiducial_left = 2 * select_0(x) - selectflip(x, knot_ind[1:])
            y_fiducial_left = selectflip(y, knot_ind[1:])
            d_fiducial_left = - selectflip(d, knot_ind[1:])
        else:
            x_fiducial_left = None
            y_fiducial_left = None
            d_fiducial_left = None

        if right in ["anti", "anti-periodic"]:
            x_fiducial_right = 2 * select_1(x) - selectflip(x, knot_ind[:-1])
            y_fiducial_right = 2 * select_1(y) - selectflip(y, knot_ind[:-1])
            d_fiducial_right = selectflip(d, knot_ind[:-1])
        elif right == "periodic":
            # first check if derivative @ boundary is zero
            if not sum(torch.index_select(d, axis, knot_ind[-1:]) == 0):
                raise Exception("Oops: derivative at periodic bc must be zero.")
            x_fiducial_right = 2 * select_1(x) - selectflip(x, knot_ind[:-1])
            y_fiducial_right = selectflip(y, knot_ind[:-1])
            d_fiducial_right = - selectflip(d, knot_ind[:-1])
        else:
            x_fiducial_right = None
            y_fiducial_right = None
            d_fiducial_right = None

        self.knots_x = self.cat([x_fiducial_left, x, x_fiducial_right], axis)
        self.knots_y = self.cat([y_fiducial_left, y, y_fiducial_right], axis)
        self.knots_d = self.cat([d_fiducial_left, d, d_fiducial_right], axis)

    @staticmethod
    def cat(catlist, axis):
        # drops the items that are `None`, and then concatenates the rest
        # if there are more than one non-`None` elements in catlist,
        # otherwise returns the one non-`None` element.
        catlist = [t for t in catlist if t is not None]
        return torch.cat(catlist, axis) if len(catlist) > 1 else catlist[0]


# =============================================================================
def test_pade22(knots_len=4, test_pade11=False, smooth=True, **kwargs):
    # To test the periodic boundary condition, you can e.g. write
    # test_pade22(extrap_left='periodic', knots_d=torch.zeros(5, 4))

    spline_kwargs = dict(extrap_left='anti', extrap_right='linear')
    spline_kwargs.update(kwargs)

    knots_x = torch.sort(torch.rand((5, knots_len))).values
    knots_y = torch.sort(torch.rand((5, knots_len))).values
    knots_d = None if smooth else torch.sort(torch.rand((5, knots_len))).values

    spline_kwargs.update(dict(knots_x=knots_x, knots_y=knots_y, knots_d=knots_d))
    x = torch.sort(torch.rand((5, 1000))).values * 2 - 0.5
    spline = Pade22Spline(**spline_kwargs)
    y = spline(x)

    import matplotlib.pyplot as plt
    color = ['b', 'r', 'g', 'm', 'c']
    for n in range(5): plt.plot(x.to('cpu')[n], y.to('cpu')[n], color=color[n])
    for n in range(5): plt.plot(knots_x.to('cpu')[n], knots_y.to('cpu')[n], 's', color=color[n])

    if test_pade11:
        y = Pade11Spline(**spline_kwargs)(x)
        for n in range(5): plt.plot(x[n].to('cpu'), y[n].to('cpu'), ':', color=color[n])

    plt.xlim([-0.5, 1.5])
    plt.ylim([-1.5, 2.5])
    plt.show()

    return spline
