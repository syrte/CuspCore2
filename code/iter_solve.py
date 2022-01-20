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


def pad3d(x):
    return np.stack([x, x * 0, x * 0], axis=-1)


def enclosedMass(pot, r):
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
        return pot, param
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
        Us_lnr_func = CubicSpline(lnr, Us, extrapolate=True)
        lnr_Us_func = CubicSpline(Us, lnr, extrapolate=True)
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

        self.lnrcir_lnrmax_func = CubicSpline(lnrmax, lnrcir, extrapolate=True)
        self.lnrmax_lnrcir_func = CubicSpline(lnrcir, lnrmax, extrapolate=True)
        self.Es_lnrcir_func = CubicSpline(lnrcir, Es, extrapolate=True)
        self.lnrcir_Es_func = CubicSpline(Es, lnrcir, extrapolate=True)
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
    def __init__(self, lnr, y, U_lnr=None):
        """
        Interpolator for N, g, f, d2ρ_dU2

        lnr, y:  array
            Using log(y) as cubic spline function of log(r).
        U_lnr:  interp_U_lnr instance, optional
            Potential.
        """
        assert np.all(np.isfinite(y))  # XXX
        assert np.all(y > 0)
        lny = np.log(y)
        set_attrs(self, locals(), excl=['self'])

    def _init_interp(self):
        self._lny_lnr_func = CubicSpline(self.lnr, self.lny, extrapolate=True)

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


def compute_f(lnr, U_lnr, ρ_U, quad, lnr_max=None):
    """
    U_lnr, d2ρdU2_lnr: interp_y_lnr object
        U, d^2ρ/dU^2
    quad: GausQuad object
    edge:
        Margine level for integration.
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


def compute_g(lnr, U_lnr, quad, lnr_min=None):
    lnr_min = lnr[0] - np.log(1e3) if lnr_min is None else lnr_min

    xi = quad.x0.reshape(-1, 1) * (lnr - lnr_min) + lnr_min
    wi = quad.w0.reshape(-1, 1) * (lnr - lnr_min)

    integ = np.exp(xi)**3 * (U_lnr(lnr) - U_lnr(xi))**0.5
    g = 16 * 2**0.5 * np.pi**2 * (integ.clip(0) * wi).sum(0)

    return interp_y_lnr(lnr, g, U_lnr=U_lnr)


def compute_N_var(lnr, f_lnr_old, U_lnr, dU_lnr, quad, lnr_min=None):
    """
    f_lnr_old: 
        function in old potential
    U_lnr, du_lnr: 
        function in new potential
    """
    lnr_min = lnr[0] - np.log(1e3) if lnr_min is None else lnr_min

    xi = quad.x0.reshape(-1, 1) * (lnr - lnr_min) + lnr_min
    wi = quad.w0.reshape(-1, 1) * (lnr - lnr_min)

    E_old = U_lnr(lnr) - dU_lnr(xi)
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

    return interp_y_lnr(lnr, M, U_lnr=U_lnr)


def compute_M(lnr, ρ_lnr, quad, lnr_cen=None, lnr_min=None, ):
    lnr_cen = lnr[0] - np.log(1e6) if lnr_cen is None else lnr_cen
    lnr_min = lnr[0] - np.log(1e3) if lnr_min is None else lnr_min
    x = np.hstack([lnr_cen, lnr_min, lnr])

    xi = quad.x0.reshape(-1, 1) * (x[1:] - x[:-1]) + x[:-1]
    wi = quad.w0.reshape(-1, 1) * (x[1:] - x[:-1])

    integ = ρ_lnr(xi) * np.exp(xi)**3
    dM = 4 * np.pi * (integ.clip(0) * wi).sum(0)
    M = np.cumsum(dM) + 4 * np.pi / 3 * np.exp(lnr_cen)**3 * ρ_lnr(lnr_cen)

    return interp_y_lnr(lnr, M[1:], U_lnr=None)


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


def _truncate_radius(r, ρ, tol=10):
    "truncate the radius where density is too close to zero"
    ρ_min = ρ[ρ > 0].min() * tol

    if ρ[0] <= ρ_min:
        # XXX: possible when rmax is not large enough
        raise ValueError('Unexpected: Found zero density (hole) at center!')

    ix = np.where(ρ <= ρ_min)[0]
    if len(ix):
        return r[:ix[0]]
    else:
        return r  # unchanged


# def _interp_y_E(E, y, tol=1e-10):
#     ix = np.where(np.diff(E) > -tol * E[:-1])[0]
#     if len(ix):
#         i1, i2 = ix[0], ix[-1] + 1
#     else:
#         i1, i2 = 0, len(E)
#     i3 = np.isfinite(y[i1:i2])  # XXX: careful!
#     return CubicSpline(E[i1:i2][i3], y[i1:i2][i3], extrapolate=True)


def _interp_y_E(E, y, U_min, tol=1e-10):
    ix = (E - U_min > -tol * U_min) & np.isfinite(y)
    return CubicSpline(E[ix], y[ix], extrapolate=True)


class IterSolver:
    # Note: any changes of the constants must be made before calling IterSolver!

    def __init__(self, den0_d, den0_g, den1_g,
                 r=np.geomspace(1e-6, 1e2, 2000),
                 rini=np.geomspace(1e-9, 1e4, 2000),
                 rcen=1e-12, nquad=100, G=1):
        """
        Initial state:
            den0_d in pot0_d+pot0_g
        Transitional:
            den0_d in pot0_d+pot1_g, with a sudden potential change
        Final state:
            den2_d, to be solved iteratively

        den0_d, pot0_g, pot1_g: agama.Potential
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
        rmin, rmax = rini[0], rini[-1]
        rcen = min(rmin * 1e-2, rcen)

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
        # we need a table covering rmin to rmax rather than r
        r_ = rini
        x_ = pad3d(r_)

        # truncate the grid inwards if the density drops to zero
        ρ0_d = den0_d.density(x_)
        r_ = _truncate_radius(r_, ρ0_d, tol=10)
        x_ = pad3d(r_)
        # rmax = r_[-1]  # XXX: careful! we'd better not update rmax
        # assert rmax > r[-1], f"Please try a grid with smaller r[-1]<{rmax:.3e}"

        ρ0_d = den0_d.density(x_)
        M0_d = den0_d.enclosedMass(r_)
        U0 = pot0.potential(x_)
        U1 = pot1.potential(x_)
        dU = U1 - U0

        lnr_ = np.log(r_)
        U0_lnr = interp_U_lnr(lnr_, U0, U0_min, G=G)
        U1_lnr = interp_U_lnr(lnr_, U1, U1_min, G=G)
        dU_lnr = CubicSpline(lnr_, dU)  # need cover rmin for integrating N1
        ρ0_U0_d = _interp_y_E(U0, ρ0_d, U0_min)  # need cover rmin for integrating f0

        ρ0_lnr_d = interp_y_lnr(lnr_, ρ0_d)  # note den0_ext is not included
        M0_lnr_d = interp_y_lnr(lnr_, M0_d)  # note den0_ext is not included

        del r_, x_, lnr_, ρ0_d, M0_d, U0, U1, dU  # clean

        # calculate f0 in U0 and new N1 in U1
        # ---------------------------------------------
        quad = GausQuad(nquad)
        lnr, lnr_min, lnr_max, lnr_cen = np.log(r), np.log(rmin), np.log(rmax), np.log(rcen)

        f0_lnr = compute_f(lnr, U0_lnr, ρ0_U0_d, quad, lnr_max)
        N1_lnr = compute_N_var(lnr, f0_lnr, U1_lnr, dU_lnr, quad, lnr_min)

        # ---------------------------------------------
        if True:
            # for self-consistency check
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
        stat = dict(U_lnr=U1_lnr, N_lnr=N1_lnr, f_lnr=f0_lnr,
                    ρ_lnr_d=ρ0_lnr_d, M_lnr_d=M0_lnr_d, ρ_d=ρ0_d, M_d=M0_d,
                    U_lnr_new=U1_lnr,  # Unew corresponds to M and ρ
                    dU_E=CubicSpline(U1_lnr(lnr), lnr * 0),
                    dU_lnr=CubicSpline(lnr, lnr * 0),
                    )
        self.stats = [stat]

    def new_potential(self, U2_lnr=None, fac_iter=1, mode='first', fac_rcir=0.8):
        """
        pot2 can be:
            agama.Potential
            potential function
            array of potential values at r
        mode: 'first', 'final', 'first+final', 'last', 
        """
        r, lnr = self.r, self.lnr
        lnr_min, lnr_max, lnr_cen = self.lnr_min, self.lnr_max, self.lnr_cen
        quad = self.quad

        # ---------------------------------------------
        if U2_lnr is None:
            U2_lnr = self.stats[-1]['U_lnr_new']

        if mode == 'first':
            U1_lnr = self.stats[0]['U_lnr'].prepare_Ecir()
            N1_lnr = self.stats[0]['N_lnr']
            lnrcir = U1_lnr.lnrcir_lnrmax_func(lnr)  # in Ui
        elif mode == 'final':
            U1_lnr = self.stats[0]['U_lnr']
            N1_lnr = self.stats[0]['N_lnr']
            Uf_lnr = self.stats[-1]['U_lnr'].prepare_Ecir()
            # dU_lnr = self.stats[-1]['dU_lnr']
            dU_E1 = self.stats[-1]['dU_E']
            E1 = U1_lnr(lnr)
            # Ef = E1 + dU_lnr(lnr)
            Ef = E1 + dU_E1(E1)
            lnrcir = Uf_lnr.lnrcir_E_func(Ef)  # in Uf
        elif mode == 'first+final':
            U1_lnr = self.stats[0]['U_lnr'].prepare_Ecir()
            N1_lnr = self.stats[0]['N_lnr']
            Uf_lnr = self.stats[-1]['U_lnr'].prepare_Ecir()
            # dU_lnr = self.stats[-1]['dU_lnr']
            dU_E1 = self.stats[-1]['dU_E']
            E1 = U1_lnr(lnr)
            # Ef = E1 + dU_lnr(lnr)
            Ef = E1 + dU_E1(E1)
            lnrcir1 = U1_lnr.lnrcir_E_func(E1) + np.log(fac_rcir)
            lnrcirf = Uf_lnr.lnrcir_E_func(Ef) + np.log(1 - fac_rcir)
            lnrcir = np.logaddexp(lnrcir1, lnrcirf)
        elif mode == 'last':
            U1_lnr = self.stats[-1]['U_lnr'].prepare_Ecir()
            N1_lnr = self.stats[-1]['N_lnr']
            lnrcir = U1_lnr.lnrcir_lnrmax_func(lnr)
        else:
            raise ValueError

        E1 = U1_lnr(lnr)
        dU = U2_lnr(lnrcir) - U1_lnr(lnrcir)  # dU(r)
        E2 = E1 + dU
        with np.errstate(invalid='ignore'):
            lnr2 = U2_lnr.lnrmax_E_func(E2)  # supress the error with E2>0

        dU_E1 = _interp_y_E(E1, dU, U1_lnr.U_min)   # what's the problem?, E1 can be ?
        dU_E1_der1 = dU_E1.derivative(1)(E1)
        # dU_lnr = CubicSpline(lnr, dU)
        # dU_E1_der1 = dU_lnr.derivative(1)(lnr) * U1_lnr.der1(lnr)

        # ---------------------------------------------
        N1 = N1_lnr(E=E1)
        N2 = N1 / (1 + dU_E1_der1)

        ix = lnr2 < lnr_max  # might get nan
        if True:
            assert np.all(np.diff(lnr2[ix]) > 0)
        N2_lnr = interp_y_lnr(lnr2[ix], N2[ix], U_lnr=U2_lnr)

        g2_lnr = compute_g(lnr, U2_lnr, quad, lnr_min)
        f2_lnr = make_f_lnr(N2_lnr, g2_lnr)

        # temporary
        ρ2_lnr_d_ = compute_ρ(lnr, f2_lnr, U2_lnr, quad, lnr_max)
        M2_lnr_d_ = compute_M(lnr, ρ2_lnr_d_, quad, lnr_cen, lnr_min)
        # M2_d_ = compute_M(lnr, f2_lnr, U2_lnr, quad, lnr_min, lnr_max).y # bug

        # ---------------------------------------------
        # weighted update
        ρ2_d = ρ2_lnr_d_.y * fac_iter + self.stats[-1]['ρ_d'] * (1 - fac_iter)
        M2_d = M2_lnr_d_.y * fac_iter + self.stats[-1]['M_d'] * (1 - fac_iter)

        ρ2_lnr_d = interp_y_lnr(lnr, ρ2_d)
        M2_lnr_d = interp_y_lnr(lnr, M2_d)
        M2_lnr = interp_y_lnr(lnr, M2_d + self.M1_g)
        U3_lnr = compute_U(lnr, M2_lnr, quad, lnr_cen, lnr_min, lnr_max, G=self.G)

        # ---------------------------------------------
        stat = dict(U_lnr=U2_lnr, N_lnr=N2_lnr, f_lnr=f2_lnr,
                    ρ_lnr_d=ρ2_lnr_d, M_lnr_d=M2_lnr_d, ρ_d=ρ2_d, M_d=M2_d,
                    U_lnr_new=U3_lnr,  # Unew corresponds to M and ρ
                    dU_E=dU_E1,
                    # dU_lnr=dU_lnr,
                    )
        self.stats.append(stat)

        return obj_attrs(locals(), excl=['self', 'ix'])

    def iter_solve(self, fac_iter=0.5, mode='first', fac_rcir=0.8, niter=50, rtol=1e-3, atol=0):
        for i in range(niter):
            res = self.new_potential(fac_iter=fac_iter, mode=mode, fac_rcir=fac_rcir)
            is_ok = np.allclose(self.stats[-1]['M_d'], self.stats[-2]['M_d'], rtol=rtol, atol=atol)
            if is_ok:
                print(f'Converged at {i}-th iteration!')
                break
        else:
            print(f'Warning: not converged within {niter} iterations!')
        return res
