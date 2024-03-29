"""
author: Zhaozhou Li (lizz.atro@gmail.com)

TOǁDO:
"""

import numpy as np
from scipy.special import roots_jacobi, roots_legendre
from scipy.interpolate import CubicSpline
import agama


def set_attrs(self, vars, excl=['self']):
    """
    vars: dict-like object
    excl: list
    """
    for key in vars:
        if key not in excl:
            setattr(self, key, vars[key])


class obj_attrs:
    def __init__(self, vars, excl=['self']):
        """
        vars: dict-like object
        excl: list
        """
        set_attrs(self, vars, excl=excl)


class CubicSplineExtrap(CubicSpline):
    def __init__(self, x, y, bc_type='not-a-knot', extrapolate='linear'):
        """
        Linearly extrapolate outside the range

        extrapolate: False, float, 'const', 'linear', 'cubic', or a 2-tuple of them

        Example
        -------
        from scipy.interpolate import PchipInterpolator, CubicSpline
        x = np.linspace(-0.7, 1, 11)
        a = np.linspace(-1.5, 2, 100)
        y = np.sin(x * pi)

        f0 = CubicSplineExtrap(x, y, extrapolate=('linear', 'const'))
        f1 = CubicSpline(x, y)
        f2 = PchipInterpolator(x, y)

        plt.figure(figsize=(8, 4))

        plt.subplot(121)
        plt.scatter(x, y)
        for i, f in enumerate([f0, f1, f2]):
            plt.plot(a, f(a), ls=['-', '--', ':'][i])
        plt.ylim(-2, 2)

        plt.subplot(122)
        for i, f in enumerate([f0, f1, f2]):
            plt.plot(a, f(a, nu=1) / np.pi, ls=['-', '--', ':'][i])
        plt.ylim(-2, 2)
        """
        if extrapolate is False:
            super().__init__(x, y, bc_type=bc_type, extrapolate=False)
        else:
            super().__init__(x, y, bc_type=bc_type, extrapolate=True)

            if np.isscalar(extrapolate):
                extrapolate = (extrapolate, extrapolate)

            xs, cs = [self.x], [self.c]
            for i, ext in enumerate(extrapolate[:2]):
                if i == 0:
                    xi, yi = x[0], y[0]
                else:
                    xi, yi = x[-1], y[-1]

                if ext == 'cubic':
                    continue
                elif ext == 'linear':
                    di = self(xi, nu=1)  # derivative at xi
                    ci = np.array([[0, 0, di, yi]]).T
                elif ext == 'const':
                    ci = np.array([[0, 0, 0, yi]]).T
                else:
                    ci = np.array([[0, 0, 0, float(ext)]]).T

                if i == 0:
                    xs, cs = [xi, *xs], [ci, *cs]
                else:
                    xs, cs = [*xs, xi], [*cs, ci]

            if len(xs) > 1:
                self.x, self.c = np.hstack(xs), np.hstack(cs)


def extend_linspace(r, rmin, rmax):
    """
    extend given linspace to cover [rmin, rmax].

    Example
    -------
    r = np.linspace(-6, 1, 1001)
    rmin, rmax = -8, 2
    a = extend_linspace(r, rmin, rmax)
    assert np.allclose(np.diff(a).mean(), np.diff(r).mean())
    assert np.allclose((a), np.linspace(*(a[[0, -1]]), len(a)), rtol=1e-10, atol=1e-20)
    assert a[0] - rmin <= 1e-10 and a[-1] - rmax >= -1e-10
    """
    dr = np.diff(r).mean()
    a0 = np.arange(r[0] - dr, rmin - dr, -dr)[::-1]
    a1 = np.arange(r[-1] + dr, rmax + dr, dr)
    return np.hstack([a0, r, a1])


def extend_geomspace(r, rmin, rmax):
    """
    extend given geomspace to cover [rmin, rmax].

    Example
    -------
    r = np.geomspace(1e-6, 1e1, 1001)
    rmin, rmax = 1e-8, 1e2
    a = extend_geomspace(r, rmin, rmax)
    assert np.allclose(np.diff(np.log(a)).mean(), np.diff(np.log(r)).mean())
    assert np.allclose((a), np.geomspace(*(a[[0, -1]]), len(a)), rtol=1e-10, atol=1e-20)
    assert a[0] - rmin <= 1e-10 and a[-1] - rmax >= -1e-10
    """
    return np.exp(extend_linspace(np.log(r), np.log(rmin), np.log(rmax)))


def pad3d(x):
    return np.stack([x, x * 0, x * 0], axis=-1)


def enclosedMass(pot, r):
    """
    equiv to pot.enclosedMass(pad3d(r))

    pot: agama.Potential
    r: float or array
    """
    return -pot.force(pad3d(r)).T[0] * r**2


def zhao_prof(M=1.0, R=None, rs=1, inner=1.0, outer=3.5, eta=0.5, rt=np.inf,
              ret_pot=False, rmin=1e-6, rmax=1e2, ngrid=500):
    """
    Make profile with amplitude M(<R)=M.
    """
    mass = M
    for i in range(2):
        param = dict(
            type='Spheroid',
            mass=mass, scaleRadius=rs,
            alpha=eta, beta=outer, gamma=inner,
            outerCutoffRadius=rt, cutoffStrength=2,
        )
        den = agama.Density(**param)

        if R is None or M == 0:
            break
        else:
            mass = mass * (M / den.enclosedMass(R))

    if ret_pot:
        pot = agama.Potential(
            type='Multipole', density=den,
            rmin=rmin, rmax=rmax, gridSizeR=ngrid, lmax=0, symmetry="s"
        )
        return den, pot, param
    else:
        return den, param


class interp_U_lnr:
    def Es_E_func(self, E, U_min=None):
        "Convert E to Es for interpolation."
        if U_min is None:
            U_min = self.U_min
        return np.log(1 / U_min - 1 / E)

    def E_Es_func(self, Es, U_min=None):
        "Convert Es back to E."
        if U_min is None:
            U_min = self.U_min
        return (1 / U_min - np.exp(Es))**-1

    def __init__(self, lnr, U, U_min, G=1, tol=1e-10):
        """
        Interpolating Us=log(1/U_min - 1/U) on log(r),
        suggested by Agama.
        Us is varying from -inf to +inf.
        """
        ix = (U - U_min > -tol * U_min)  # XXX: need to be more careful!
        lnr, U = lnr[ix], U[ix]

        Us = self.Es_E_func(U, U_min)
        Us_lnr_func = CubicSplineExtrap(lnr, Us, extrapolate='linear')
        lnr_Us_func = CubicSplineExtrap(Us, lnr, extrapolate='linear')
        Us_lnr_der1 = Us_lnr_func.derivative(1)

        r = np.exp(lnr)
        set_attrs(self, locals(), excl=['self', 'ix'])

    def __call__(self, lnr=None, r=None):
        """
        return U(lnr) or U(r) depending on inputs
        """
        if lnr is None:
            lnr = np.log(r)
        return self.U_lnr_func(lnr)

    def U_lnr_func(self, lnr):
        "U(lnr)"
        Us = self.Us_lnr_func(lnr)
        return self.E_Es_func(Us)

    def U_lnr_der1(self, lnr):
        "dU/dlnr"
        U = self.U_lnr_func(lnr)
        return U * (U / self.U_min - 1) * self.Us_lnr_der1(lnr)

    der1 = U_lnr_der1

    def M_lnr_func(self, lnr):
        "M(lnr) = r / G * dU/dlnr"
        return np.exp(lnr) / self.G * self.U_lnr_der1(lnr)

    def lnr_U_func(self, U):
        "lnr(U)"
        Us = self.Es_E_func(U)
        return self.lnr_Us_func(Us)

    def lnrmax_E_func(self, E):
        "lnrmax(E), where rmax(E) is the solution of U(rmax)=E"
        return self.lnr_U_func(E)

    def r_U_func(self, U):
        "r(U)"
        lnr = self.lnr_U_func(U)
        return np.exp(lnr)

    rmax_E_func = r_U_func

    def Ecir_lnr(self, lnr):
        Mcir = self.M_lnr_func(lnr)
        Ucir = self.U_lnr_func(lnr)
        rcir = np.exp(lnr)
        Ecir = 0.5 * self.G * Mcir / rcir + Ucir
        return Ecir

    def prepare_Ecir(self):
        "Call this first bebore using Ecir related functions"
        lnrcir = self.lnr
        Ecir = self.Ecir_lnr(lnrcir)
        Es = self.Es_E_func(Ecir)
        lnrmax = self.lnr_Us_func(Es)

        self.lnrcir_lnrmax_func = CubicSplineExtrap(lnrmax, lnrcir, extrapolate='linear')
        self.lnrmax_lnrcir_func = CubicSplineExtrap(lnrcir, lnrmax, extrapolate='linear')
        self.Es_lnrcir_func = CubicSplineExtrap(lnrcir, Es, extrapolate='linear')
        self.lnrcir_Es_func = CubicSplineExtrap(Es, lnrcir, extrapolate='linear')
        return self

    def Ecir_lnr_func(self, lnr):
        Es = self.Es_lnrcir_func(lnr)
        return self.E_Es_func(Es)

    def lnrcir_E_func(self, E):
        Es = self.Es_E_func(E)
        return self.lnrcir_Es_func(Es)

    def rcir_E_func(self, E):
        Es = self.Es_E_func(E)
        lnrcir = self.lnrcir_Es_func(Es)
        return np.exp(lnrcir)


class interp_y_lnr:
    def __init__(self, lnr, y, U_lnr=None, extrapolate='linear', clip=False):
        """
        Interpolator for N, g, f, d2ρ_dU2

        lnr, y:  array
            Using log(y) as cubic spline function of log(r).
        U_lnr:  interp_U_lnr instance, optional
            Potential, requred if want to call with energy.
        clip:
            Truncate lnr with y<=0. Should use it only with known y=0.
        """
        assert np.all(np.isfinite(y))  # any nan, inf?

        if clip:
            ix = y > 0
            lnr, y = lnr[ix], y[ix]
        else:
            assert np.all(y > 0)

        lny = np.log(y)
        set_attrs(self, locals(), excl=['self'])

    def _init_interp(self):
        self._lny_lnr_func = CubicSplineExtrap(
            self.lnr, self.lny, extrapolate=self.extrapolate)

    def __call__(self, lnr=None, r=None, E=None):
        "Return fn(lnr), fn(r), or fn(E) depending on kwargs"
        if not hasattr(self, '_lny_lnr_func'):
            # delayed creation of interpolator
            self._init_interp()

        if lnr is not None:
            lny = self._lny_lnr_func(lnr)
            return np.exp(lny)
        elif r is not None:
            lnr = np.log(r)
            return self(lnr)
        elif E is not None:
            lnr = self.U_lnr.lnr_U_func(E)
            return self(lnr)
        else:
            raise ValueError


def _interp_y_E(E, y, U_min, tol=1e-10):
    ix = (E - U_min > -tol * U_min) & np.isfinite(y)   # XXX: careful
    return CubicSpline(E[ix], y[ix], extrapolate=True)


class make_f_lnr:
    def __init__(self, N_lnr, g_lnr):
        self.N_lnr, self.g_lnr = N_lnr, g_lnr

    def __call__(self, *args, **kwargs):
        "Return f(lnr), f(r), or f(E) depending on kwargs"
        return self.N_lnr(*args, **kwargs) / self.g_lnr(*args, **kwargs)


class make_N_lnr:
    def __init__(self, f_lnr, g_lnr):
        self.f_lnr, self.g_lnr = f_lnr, g_lnr

    def __call__(self, *args, **kwargs):
        "Return N(lnr), N(r), or N(E) depending on kwargs"
        return self.f_lnr(*args, **kwargs) * self.g_lnr(*args, **kwargs)


class GausQuad:
    def __init__(self, n):
        """Prepare tables for Gauss–Jacobi quadrature grid
        """
        x0, w0 = roots_legendre(n)  # ∫ f(x) dx
        x1, w1 = roots_jacobi(n, -0.5, 0)  # ∫ f(x) / sqrt(a - x) dx
        x2, w2 = roots_jacobi(n, 0, -0.5)  # ∫ f(x) / sqrt(x - a) dx
        x3, w3 = roots_jacobi(n, +0.5, 0)  # ∫ f(x) * sqrt(a - x) dx
        x4, w4 = roots_jacobi(n, 0, +0.5)  # ∫ f(x) * sqrt(x - a) dx
        x0, w0 = x0 * 0.5 + 0.5, w0 * 0.5
        x1, w1 = x1 * 0.5 + 0.5, w1 * 0.5**0.5
        x2, w2 = x2 * 0.5 + 0.5, w2 * 0.5**0.5
        x3, w3 = x3 * 0.5 + 0.5, w3 * 0.5**1.5
        x4, w4 = x4 * 0.5 + 0.5, w4 * 0.5**1.5
        set_attrs(self, locals(), excl=['self'])


def compute_f_old(lnr, U_lnr, ρ_U, quad, lnr_max=None):
    """
    Obsolete: ρ_U is not as stable as d2ρdUdlnr_lnr

    U_lnr, d2ρdU2_lnr: interp_y_lnr object
        U, d^2ρ/dU^2
    quad: GausQuad object
    """
    lnr_max = lnr[-1] + np.log(1e3) if lnr_max is None else lnr_max

    xi = quad.x2.reshape(-1, 1) * (lnr_max - lnr) + lnr
    wi = quad.w2.reshape(-1, 1) * (lnr_max - lnr)**0.5
    Ui = U_lnr(xi)
    d2ρ_dU2 = ρ_U.derivative(2)(Ui)

    integ = ((xi - lnr) / (Ui - U_lnr(lnr)))**0.5 * d2ρ_dU2 * U_lnr.der1(xi)
    f = 8**-0.5 * np.pi**-2 * (integ.clip(0) * wi).sum(0)
    # ignore the boundary term, which is correct at least
    # for any density profile with outer slope steeper than -1

    return interp_y_lnr(lnr, f, U_lnr=U_lnr)


def compute_f(lnr, U_lnr, d2ρdUdlnr_lnr, quad, lnr_max=None):
    lnr_max = lnr[-1] + np.log(1e3) if lnr_max is None else lnr_max

    xi = quad.x2.reshape(-1, 1) * (lnr_max - lnr) + lnr
    wi = quad.w2.reshape(-1, 1) * (lnr_max - lnr)**0.5
    d2ρ_dU2 = d2ρdUdlnr_lnr(xi)

    integ = ((xi - lnr) / (U_lnr(xi) - U_lnr(lnr)))**0.5 * d2ρ_dU2
    # XXX: should we clip integ with 0?
    f = 8**-0.5 * np.pi**-2 * (integ * wi).sum(0)
    # ignore the boundary term, which is correct at least
    # for any density profile with outer slope steeper than -1

    return interp_y_lnr(lnr, f, U_lnr=U_lnr)


def compute_g(lnr, U_lnr, quad, lnr_min=None):
    lnr_min = lnr[0] - np.log(1e3) if lnr_min is None else lnr_min

    xi = quad.x0.reshape(-1, 1) * (lnr - lnr_min) + lnr_min
    wi = quad.w0.reshape(-1, 1) * (lnr - lnr_min)

    integ = np.exp(xi)**3 * (U_lnr(lnr) - U_lnr(xi))**0.5
    g = 16 * 2**0.5 * np.pi**2 * (integ.clip(0) * wi).sum(0)

    return interp_y_lnr(lnr, g, U_lnr=U_lnr)


def compute_N_var(lnr, f_lnr_old, U_lnr_old, U_lnr, quad, lnr_min=None):
    """
    f_lnr_old, U_lnr_old: 
        function in old potential
    U_lnr:
        function in new potential
    """
    lnr_min = lnr[0] - np.log(1e3) if lnr_min is None else lnr_min

    xi = quad.x0.reshape(-1, 1) * (lnr - lnr_min) + lnr_min
    wi = quad.w0.reshape(-1, 1) * (lnr - lnr_min)

    E_old = U_lnr(lnr) + (U_lnr_old(xi) - U_lnr(xi))

    # truncate E_old to the range of old potential, important for deepened potentials (eta>0)
    E_min = (1 - np.finfo('f8').eps) * f_lnr_old.U_lnr.U_min  # 1-2e-16
    E_max = -np.finfo('f8').tiny  # -2e-308
    E_old[(E_old <= E_min) | (E_old >= E_max)] = E_max  # set outliers with f(E) = 0

    integ = np.exp(xi)**3 * (U_lnr(lnr) - U_lnr(xi))**0.5 * f_lnr_old(E=E_old)
    N = 16 * 2**0.5 * np.pi**2 * (integ.clip(0) * wi).sum(0)

    return interp_y_lnr(lnr, N, U_lnr=U_lnr)


def compute_ρ(lnr, f_lnr, U_lnr, quad, lnr_max=None):
    lnr_max = lnr[-1] + np.log(1e3) if lnr_max is None else lnr_max

    xi = quad.x0.reshape(-1, 1) * (lnr_max - lnr) + lnr
    wi = quad.w0.reshape(-1, 1) * (lnr_max - lnr)

    integ = f_lnr(xi) * (U_lnr(xi) - U_lnr(lnr))**0.5 * U_lnr.der1(xi)
    ρ = 4 * 2**0.5 * np.pi * (integ.clip(0) * wi).sum(0)

    return interp_y_lnr(lnr, ρ, U_lnr=U_lnr)


def compute_M_f(lnr, f_lnr, U_lnr, quad, lnr_min=None, lnr_max=None):
    "Alternative way for calcute M. Expected to be more precise than compute_M from ρ?"
    # XXX: a bug to be resolve.
    lnr_min = lnr[0] - np.log(1e3) if lnr_min is None else lnr_min
    lnr_max = lnr[-1] + np.log(1e3) if lnr_max is None else lnr_max

    tj = quad.x0.reshape(1, -1, 1) * (lnr_max - lnr_min) + lnr_min
    wj = quad.w0.reshape(1, -1, 1) * (lnr_max - lnr_min)

    xi = quad.x0.reshape(-1, 1, 1) * (np.fmin(lnr, tj) - lnr_min) + lnr_min
    wi = quad.w0.reshape(-1, 1, 1) * (np.fmin(lnr, tj) - lnr_min) * wj

    integ = np.exp(xi)**3 * f_lnr(tj) * (U_lnr(tj) - U_lnr(xi))**0.5 * U_lnr.der1(tj)
    M = 16 * 2**0.5 * np.pi**2 * (integ.clip(0) * wi).reshape(-1, len(lnr)).sum(0)

    return interp_y_lnr(lnr, M, U_lnr=U_lnr, extrapolate=('linear', 'const'))


def compute_M(lnr, ρ_lnr, quad, lnr_cen=None, lnr_min=None):
    lnr_cen = lnr[0] - np.log(1e6) if lnr_cen is None else lnr_cen
    lnr_min = lnr[0] - np.log(1e3) if lnr_min is None else lnr_min
    x = np.hstack([lnr_cen, lnr_min, lnr])

    xi = quad.x0.reshape(-1, 1) * (x[1:] - x[:-1]) + x[:-1]
    wi = quad.w0.reshape(-1, 1) * (x[1:] - x[:-1])

    integ = ρ_lnr(xi) * np.exp(xi)**3
    dM = 4 * np.pi * (integ.clip(0) * wi).sum(0)
    M = np.cumsum(dM) + 4 * np.pi / 3 * np.exp(lnr_cen)**3 * ρ_lnr(lnr_cen)

    return interp_y_lnr(lnr, M[1:], U_lnr=None, extrapolate=('linear', 'const'))
    # we don't want mass to increase all the way, use const extrap on the right
    # XXX: but it may break the continuity of U?


def compute_U(lnr, M_lnr, quad, lnr_cen=None, lnr_min=None, lnr_max=None, G=1):
    lnr_cen = lnr[0] - np.log(1e6) if lnr_cen is None else lnr_cen
    lnr_min = lnr[0] - np.log(1e3) if lnr_min is None else lnr_min
    lnr_max = lnr[-1] + np.log(1e3) if lnr_max is None else lnr_max
    x = np.hstack([lnr_cen, lnr_min, lnr, lnr_max])
    # lnr_cen for covering region inside lnr_min

    xi = quad.x0.reshape(-1, 1) * (x[1:] - x[:-1]) + x[:-1]
    wi = quad.w0.reshape(-1, 1) * (x[1:] - x[:-1])

    integ = M_lnr(xi) / np.exp(xi)
    dU = G * (integ.clip(0) * wi).sum(0)
    U = -np.cumsum(dU[::-1])[::-1] - G * M_lnr(lnr_max) / np.exp(lnr_max)
    # to ensure U(rmax) = - G * M(rmax) / rmax

    return interp_U_lnr(lnr, U=U[2:], U_min=U[0], G=G)


def _truncate_radius(r, ρ, tol=10, ret_idx=True):
    "truncate the radius where density is too close to zero"
    ρ_min = ρ[ρ > 0].min() * tol

    if ρ[0] <= ρ_min:
        # XXX: possible when rmax is not large enough
        raise ValueError('Unexpected: Found zero density (hole) at center!')

    ix = np.where(ρ <= ρ_min)[0]
    if len(ix):
        idx = slice(None, ix[0], None)
    else:
        idx = slice(None, None, None)  # unchanged
    if ret_idx:
        return idx
    else:
        return r[idx]


class compute_d2ρdUdlnr:
    def __init__(self, lnr, ρ, M, G):
        """
        d^2ρ/dU^2 * dU/dlnr
        ρ: tracer density
        M: *total* mass
        """
        ix = ρ > 0  # truncate the grid inwards if the density drops to zero
        lnρ_lnr = CubicSplineExtrap(lnr[ix], np.log(ρ[ix]), extrapolate='linear')
        lnM_lnr = CubicSplineExtrap(lnr, np.log(M), extrapolate=('linear', 'const'))
        # we don't want mass to increase all the way, use const extrap on the right

        set_attrs(self, locals(), excl=['self', 'ix'])

    def __call__(self, lnr):
        G = self.G
        r = np.exp(lnr)
        M = np.exp(self.lnM_lnr(lnr))
        ρ = np.exp(self.lnρ_lnr(lnr))
        lnρ_der1 = self.lnρ_lnr(lnr, nu=1)  # dlnρ/dlnr
        lnρ_der2 = self.lnρ_lnr(lnr, nu=2)  # d^2lnρ/dlnr^2
        lnM_der1 = self.lnM_lnr(lnr, nu=1)  # dlnM/dlnr

        d2ρdUdlnr = (r * ρ) / (G * M) * (lnρ_der2 + lnρ_der1 * (1 + lnρ_der1 - lnM_der1))
        return d2ρdUdlnr


def none_min(x, a):
    if x is None:
        return a
    else:
        return min(a, x)


def none_max(x, a):
    if x is None:
        return a
    else:
        return max(a, x)


class IterSolver:
    # Note: any changes of the constants must be made before calling IterSolver!

    def __init__(self, den0_d, den0_g, den1_g,
                 r=np.geomspace(1e-6, 1e2, 2000),
                 rlim=None,
                 nquad=100, G=1):
        """
        Initial state:
            den0_d in pot0_d+pot0_g
        Transitional:
            den0_d in pot0_d+pot1_g, with a sudden potential change
        Final state:
            den2_d, to be solved iteratively

        den0_d, den0_g, den1_g: agama.Density
            Inital density and exteral potentials.
            The code does not expect tracer density with slope greater than -2 in the center.
        r:  array
            The radial range interested, use extrapolation outside this range.
        rmin, rmax:  float
            The radial range used for integration, should exceed r sufficiently.
        rcen:  float
            Used for calculating Umin, which is crutial for interpolating U(lnr),
            should be smaller than rmin.
        """
        # intialize
        if rlim is None:
            rcen, rmin, rmax = None, None, None
        elif len(rlim) == 2:
            rcen, rmin, rmax = [None, *rlim]
        else:
            rcen, rmin, rmax = rlim

        # ---------------------------------------------
        # prepare potentials
        rmin = none_min(rmin, r[0] * 1e-2)
        rcen = none_min(rcen, rmin * 1e-2)
        rmax = none_min(rmax, r[-1] * 1e1)

        def make_pot(den, rmin, rmax):
            return agama.Potential(
                type="Multipole", density=den, rmin=rmin, rmax=rmax,
                gridSizeR=500, lmax=0, symmetry="s"
            )
        pot0_d = make_pot(den0_d, rmin, rmax)
        pot0_g = make_pot(den0_g, rmin, rmax)
        pot1_g = make_pot(den1_g, rmin, rmax)
        pot0 = agama.Potential(pot0_d, pot0_g)
        pot1 = agama.Potential(pot0_d, pot1_g)
        U0_min = pot0.potential(pad3d(0))
        U1_min = pot1.potential(pad3d(0))

        # ---------------------------------------------
        # prepare interplators
        # we need a table covering rmin to rmax
        r_ = extend_geomspace(r, rmin, rmax)
        x_ = pad3d(r_)

        ρ0_d = den0_d.density(x_)
        M0_d = den0_d.enclosedMass(r_)
        M0 = den0_d.enclosedMass(r_) + den0_g.enclosedMass(r_)  # total mass
        U0 = pot0.potential(x_)
        U1 = pot1.potential(x_)
        dU = U1 - U0

        lnr_ = np.log(r_)
        U0_lnr = interp_U_lnr(lnr_, U0, U0_min, G=G)
        U1_lnr = interp_U_lnr(lnr_, U1, U1_min, G=G)
        dU_lnr = CubicSpline(lnr_, dU)  # need cover rmin for integrating N1
        d2ρdUdlnr0_lnr = compute_d2ρdUdlnr(lnr_, ρ0_d, M0, G)

        ρ0_lnr_d = interp_y_lnr(lnr_, ρ0_d, clip=True)  # tracer density
        M0_lnr_d = interp_y_lnr(lnr_, M0_d)

        # obsolete
        # ix = _truncate_radius(r_, ρ0_d, tol=10)  # XXX: careful
        # ρ0_U0_d = _interp_y_E(U0[ix], ρ0_d[ix], U0_min)  # need cover rmin for integrating f0
        # f0_lnr_old = compute_f_old(lnr, U0_lnr, ρ0_U0_d, quad, lnr_max)

        del r_, x_, lnr_, ρ0_d, M0_d, U0, U1, dU  # clean

        # calculate f0 in U0 and new N1 in U1
        # ---------------------------------------------
        quad = GausQuad(nquad)
        lnr, lnr_min, lnr_max, lnr_cen = np.log(r), np.log(rmin), np.log(rmax), np.log(rcen)

        f0_lnr = compute_f(lnr, U0_lnr, d2ρdUdlnr0_lnr, quad, lnr_max)
        N1_lnr = compute_N_var(lnr, f0_lnr, U0_lnr, U1_lnr, quad, lnr_min)

        # ---------------------------------------------
        if True:
            # for self-consistency check
            g1_lnr = compute_g(lnr, U1_lnr, quad, lnr_min)
            g0_lnr = compute_g(lnr, U0_lnr, quad, lnr_min)
            N0_lnr = make_N_lnr(f0_lnr, g0_lnr)
            ρ0_lnr_d_ = compute_ρ(lnr, f0_lnr, U0_lnr, quad, lnr_max)
            M0_lnr_d_ = compute_M(lnr, ρ0_lnr_d_, quad, lnr_cen, lnr_min)
            U0_lnr_d_ = compute_U(lnr, M0_lnr_d_, quad, lnr_cen, lnr_min, lnr_max, G=G)
            # M0_lnr_d_f = compute_M_f(lnr, f0_lnr, U0_lnr, quad, lnr_min, lnr_max)  # bug

        x = pad3d(r)
        ρ0_d = den0_d.density(x)
        ρ0_g = den0_g.density(x)
        ρ1_g = den1_g.density(x)
        M0_d = den0_d.enclosedMass(r)
        M0_g = den0_g.enclosedMass(r)
        M1_g = den1_g.enclosedMass(r)  # needed for calculating Mtot later

        # ---------------------------------------------
        set_attrs(self, locals(), excl=['self'])
        stat = dict(U_lnr=U1_lnr, N_lnr=N1_lnr, f_lnr=f0_lnr, g_lnr=g0_lnr,
                    ρ_lnr_d=ρ0_lnr_d, M_lnr_d=M0_lnr_d, ρ_d=ρ0_d, M_d=M0_d,
                    U_lnr_new=U1_lnr,  # Unew corresponds to M and ρ
                    dU_E=CubicSpline(U1_lnr(lnr), lnr * 0),
                    # dU_lnr=CubicSpline(lnr, lnr * 0),
                    )
        self.stats = [obj_attrs(stat)]

    def new_potential(self, U2_lnr=None, fac_iter=1, mode='first', fac_rcir=0.8):
        """
        pot2 can be:
            agama.Potential
            potential function
            array of potential values at r
        mode: 'first', 'last', 'first+last', 'iter', 
        """
        r, lnr = self.r, self.lnr
        lnr_min, lnr_max, lnr_cen = self.lnr_min, self.lnr_max, self.lnr_cen
        quad = self.quad

        # ---------------------------------------------
        if U2_lnr is None:
            U2_lnr = self.stats[-1].U_lnr_new

        if mode == 'first':
            U1_lnr = self.stats[0].U_lnr.prepare_Ecir()
            N1_lnr = self.stats[0].N_lnr
            lnrcir = U1_lnr.lnrcir_lnrmax_func(lnr)  # in Ui
        elif mode == 'last':
            U1_lnr = self.stats[0].U_lnr
            N1_lnr = self.stats[0].N_lnr
            Uf_lnr = self.stats[-1].U_lnr.prepare_Ecir()
            # dU_lnr = self.stats[-1].dU_lnr
            dU_E1 = self.stats[-1].dU_E
            E1 = U1_lnr(lnr)
            # Ef = E1 + dU_lnr(lnr)
            Ef = E1 + dU_E1(E1)
            lnrcir = Uf_lnr.lnrcir_E_func(Ef)  # in Uf
        elif mode == 'first+last':
            U1_lnr = self.stats[0].U_lnr.prepare_Ecir()
            N1_lnr = self.stats[0].N_lnr
            Uf_lnr = self.stats[-1].U_lnr.prepare_Ecir()
            # dU_lnr = self.stats[-1].dU_lnr
            dU_E1 = self.stats[-1].dU_E
            E1 = U1_lnr(lnr)
            # Ef = E1 + dU_lnr(lnr)
            Ef = E1 + dU_E1(E1)
            lnrcir1 = U1_lnr.lnrcir_E_func(E1) + np.log(fac_rcir)
            lnrcirf = Uf_lnr.lnrcir_E_func(Ef) + np.log(1 - fac_rcir)
            lnrcir = np.logaddexp(lnrcir1, lnrcirf)
        elif mode == 'iter':
            U1_lnr = self.stats[-1].U_lnr.prepare_Ecir()
            N1_lnr = self.stats[-1].N_lnr
            lnrcir = U1_lnr.lnrcir_lnrmax_func(lnr)
        else:
            raise ValueError

        E1 = U1_lnr(lnr)
        dU = U2_lnr(lnrcir) - U1_lnr(lnrcir)  # dU(r)
        E2 = E1 + dU
        with np.errstate(invalid='ignore'):
            lnr2 = U2_lnr.lnrmax_E_func(E2)  # supress the error with E2>0

        dU_E1 = _interp_y_E(E1, dU, U1_lnr.U_min)   # XXX: what's the problem?, E1 can be ?
        dU_E1_der1 = dU_E1.derivative(1)(E1)
        # dU_lnr = CubicSpline(lnr, dU)
        # dU_E1_der1 = dU_lnr.derivative(1)(lnr) * U1_lnr.der1(lnr)

        # ---------------------------------------------
        N1 = N1_lnr(E=E1)
        N2 = N1 / (1 + dU_E1_der1)

        ix = lnr2 < lnr_max  # might get nan
        if True:
            assert np.all(np.diff(lnr2[ix]) > 0)  # XXX: why sometimes fail?
        N2_lnr = interp_y_lnr(lnr2[ix], N2[ix], U_lnr=U2_lnr)

        g2_lnr = compute_g(lnr, U2_lnr, quad, lnr_min)
        f2_lnr = make_f_lnr(N2_lnr, g2_lnr)

        # temporary
        ρ2_lnr_d_ = compute_ρ(lnr, f2_lnr, U2_lnr, quad, lnr_max)
        M2_lnr_d_ = compute_M(lnr, ρ2_lnr_d_, quad, lnr_cen, lnr_min)
        # M2_d_ = compute_M(lnr, f2_lnr, U2_lnr, quad, lnr_min, lnr_max).y # bug

        # ---------------------------------------------
        # weighted update
        ρ2_d = ρ2_lnr_d_.y * fac_iter + self.stats[-1].ρ_d * (1 - fac_iter)
        M2_d = M2_lnr_d_.y * fac_iter + self.stats[-1].M_d * (1 - fac_iter)

        ρ2_lnr_d = interp_y_lnr(lnr, ρ2_d)
        M2_lnr_d = interp_y_lnr(lnr, M2_d)
        M2_lnr = interp_y_lnr(lnr, M2_d + self.M1_g)
        U3_lnr = compute_U(lnr, M2_lnr, quad, lnr_cen, lnr_min, lnr_max, G=self.G)

        # ---------------------------------------------
        stat = dict(U_lnr=U2_lnr, N_lnr=N2_lnr, f_lnr=f2_lnr, g_lnr=g2_lnr,
                    ρ_lnr_d=ρ2_lnr_d, M_lnr_d=M2_lnr_d, ρ_d=ρ2_d, M_d=M2_d,
                    U_lnr_new=U3_lnr,  # Unew corresponds to M and ρ
                    dU_E=dU_E1,
                    # dU_lnr=dU_lnr,
                    )
        self.stats.append(obj_attrs(stat))

        return obj_attrs(locals(), excl=['self', 'ix'])

    def iter_solve(self, fac_iter=0.5, mode='first', fac_rcir=0.8, niter=50, rtol=1e-3, atol=0):
        for i in range(niter):
            res = self.new_potential(fac_iter=fac_iter, mode=mode, fac_rcir=fac_rcir)
            is_ok = np.allclose(self.stats[-1].M_d, self.stats[-2].M_d, rtol=rtol, atol=atol)
            if is_ok:
                print(f'Model "{mode}": Converged at {i+1}-th iteration! ')
                break
        else:
            print(f'Model "{mode}": Not converged within {niter} iterations!')
        return res
