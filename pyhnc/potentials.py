#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This program is part of pyHNC, copyright (c) 2023 Patrick B Warren (STFC).
# Additional modifications copyright (c) 2025 Joshua F Robinson (STFC).  
# Email: patrick.warren{at}stfc.ac.uk.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <http://www.gnu.org/licenses/>.

# Hyper-netted chain (HNC) solver for Ornstein-Zernike (OZ) equation.

try: from .utilities import *
except ImportError: from utilities import *

from abc import ABC, abstractmethod
from typing import Type
from numpy.typing import NDArray
from scipy.special import erf

class Potential(ABC):

    def copy(self):
        cls = self.__class__
        new = cls(*self.__getinitargs__())
        new.__setstate__(self.__getstate__())
        return new

    def __getinitargs__(self):
        return

    def __reduce_ex__(self, protocol):
        return (
            self.__class__,
            self.__getinitargs__(),
            self.__getstate__(),
        )

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self.__dict__.update(state)

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    def subset(self, species: NDArray | list):
        """Construct potential on just subset of species"""
        raise NotImplementedError

    @property
    @abstractmethod
    def nspecies(self):
        raise NotImplementedError

    @abstractmethod
    def potential(self, r: NDArray | float):
        raise NotImplementedError

    @abstractmethod
    def force(self, r: NDArray | float):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.potential(*args, **kwargs)


class ShortRangeResidual(Potential):
    r"""Excess part of a potential after removing a long-range part.

    In solving the OZ equation it's often convenient to separate off the
    long-range tail using the asymptotic properties of correlation functions.
    In particular, $c(r) \to -\beta v(r)$ for large $r$. This can be exploited
    to use a smaller grid size and improve convergence. This class is a helper
    interface to access the remaining part of $v(r)$ after subtracting the
    long-range part that $v(r)$ approaches at large $r$.
    """

    def __repr__(self):
        return rf'<{type(self).__name__} full={self.full} long={self.full.long}>'

    def __init__(self, full):
        self.full = full
        assert self.full.nspecies == self.full.long.nspecies

    def __eq__(self, other):
        if type(other) is not type(self): return False
        return self.full == other.full

    @property
    def nspecies(self):
        assert self.full.nspecies == self.full.long.nspecies
        return self.full.nspecies

    def potential(self, r: float | NDArray):
        return self.full.potential(r) - self.full.long.potential(r)

    def force(self, r: float | NDArray):
        return self.full.force(r) - self.full.long.force(r)


class DPD(Potential):
    r"""Quadratic potential used for coarse-graining in dissipative particle
    dynamics (DPD):

        $$v(r) = \frac{A}{2} (1 - r/r_c)^2 \qquad \forall r \le \rcut\,,$$

    and $v(r) = 0$ for $r > r_c$. This potential is convenient as the force
    decreases linearly from $r = 0$ to $\rcut$ making it very soft and thus
    suitable for large time-steps.
    """

    def __getinitargs__(self):
        return self.A.copy(), self.rcut.copy()

    def __repr__(self):
        return rf'<{type(self).__name__} A={self.A.tolist()} rc={self.rcut.tolist()}>'

    def __init__(self, A: float | NDArray,
                 rcut: float | NDArray=1.):
        A = np.atleast_2d(A)
        assert A.shape[0] == A.shape[1]
        rcut = np.atleast_2d(rcut)
        assert rcut.shape[0] == rcut.shape[1]

        if 1 in rcut.shape:
            rcut = rcut[0][0] * np.ones_like(A)

        self.A = A
        self.rcut = rcut

    def __eq__(self, other):
        if type(other) is not type(self): return False
        return np.allclose(self.A, other.A) and \
               np.allclose(self.rcut, other.rcut)

    @property
    def nspecies(self):
        return len(self.A)

    def subset(self, species: list | NDArray):
        species = np.atleast_1d(species)
        assert np.all(species >= 0) and np.all(species <= self.nspecies)
        indices = np.ix_(species, species)
        return DPD(self.A[self.indices], self.rcut[indices])

    def potential(self, r: float | NDArray):
        r = np.atleast_1d(r)

        A = np.asarray(self.A)[..., None]
        rcut = np.asarray(self.rcut)[..., None]
        r = r[None, None, :]
        v = 0.5 * A * (1 - r / rcut)**2
        v[r > rcut] = 0.

        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()
        return v

    def force(self, r: float | NDArray):
        r = np.atleast_1d(r)

        A = np.asarray(self.A)[..., None]
        rcut = np.asarray(self.rcut)[..., None]
        r = r[None, None, :]
        f = A * (1 - r / rcut) / rcut
        f[r > rcut] = 0.

        f = np.squeeze(f)
        if f.ndim == 0: f = f.item()
        return f


def test_dpd():
    import pickle
    def test_copy(v, v2):
        assert v2 is not v
        assert v == v2
        assert np.all(v.A == v2.A)
        assert np.all(v.rcut == v2.rcut)
        v2.A += 1
        v2.rcut += 1
        assert v != v2
        assert np.all(v.A != v2.A)
        assert np.all(v.rcut != v2.rcut)

    r = np.linspace(0, 10, 100)

    # Tests for single-component systems.

    A = 25
    v = DPD(A)
    assert np.isscalar(v.potential(1.))
    assert not np.isscalar(v.potential(r))
    assert v.copy().potential(1.) == v.potential(1.)
    test_copy(v, v.copy())
    test_copy(v, pickle.loads(pickle.dumps(v)))

    from scipy.optimize import approx_fprime
    exact = np.array([-approx_fprime(rr, v.potential) for rr in r]).reshape(-1)
    assert np.allclose(v.force(r), exact, rtol=1e-6)

    rcut = 2.
    v = DPD(A, rcut)
    phi = v.potential(r)
    assert np.allclose(phi[r > rcut], 0.)
    assert np.all(~np.isclose(phi[r < rcut], 0.))

    # Test it also works for binary and ternary mixtures.
    for n in [2, 3]:
        A = 25 * np.ones((n, n))
        v = DPD(A)
        assert v.potential(1.).shape == A.shape
        assert v.potential(r).shape == (n, n, r.size)
        test_copy(v, v.copy())
        test_copy(v, pickle.loads(pickle.dumps(v)))

    # Gradient test for mixtures.
    r = np.linspace(0, 2, 11)
    for n in [2, 3]:
        dA = np.random.random((n, n))
        dA = dA + dA.T
        A = 25 * np.ones((n, n)) + dA
        dR = 0.1 * np.random.random((n, n))
        dR = dR + dR.T
        R = np.ones((n,n)) + dR
        v = DPD(A, R)
        f = lambda x: v.potential(x).reshape(-1)
        exact = np.array([-approx_fprime(rr, f) for rr in r]).T.reshape(n,n,len(r))
        analytic = v.force(r)
        assert np.allclose(exact, analytic)


class GaussianIonLongRange(Potential):
    r"""Long range part of GaussianIon."""

    def __repr__(self):
        return rf'<{type(self).__name__} z={self.full.z} α={self.full.α} lB={self.full.lB}>'

    def __init__(self, full):
        self.full = full

    def __eq__(self, other):
        if type(other) is not type(self): return False
        return self.full == other.full

    @property
    def nspecies(self):
        return self.full.nspecies

    def subset(self, species: list | NDArray):
        raise RuntimeError('should never be called!')

    def potential(self, r: float | NDArray):
        r = np.atleast_1d(r)

        with np.errstate(invalid='ignore'):
            v = self.full.lB * erf(self.full.α**0.5 * r) / r
        v = np.outer(self.full.z, self.full.z)[:,:,None] * v[None,None,:]

        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()
        return v

    def potential_fourier(self, k: float | NDArray):
        k = np.atleast_1d(k)

        with np.errstate(invalid='ignore'):
            v = 4*np.pi * self.full.lB * np.exp(-k**2 / (4*self.full.α)) / k**2
        v = np.outer(self.full.z, self.full.z)[:,:,None] * v[None,None,:]

        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()
        return v

    def force(self, r: float | NDArray):
        r = np.atleast_1d(r)
        α = self.full.α

        with np.errstate(invalid='ignore'):
            f = self.full.lB * (
                    erf(α**0.5 * r)/r - 2*(α/np.pi)**0.5 * np.exp(-α*r**2)
                ) / r
        f = np.outer(self.full.z, self.full.z)[:,:,None] * f[None,None,:]

        f = np.squeeze(f)
        if f.ndim == 0: f = f.item()
        return f


class GaussianIon(Potential):
    r"""Interactions between ions with normally distributed charges:

        $$\rho_i(r) = \ell_\mathrm{B} z_i
        \left( \frac{\alpha}{\pi} \right)^{3/2} e^{-\alpha r^2}\,,$$

    where $r$ is the distance from the atom centre, $\ell_\mathrm{B}$ is the
    Bjerrum length and $z_i$ is the valence of species $i$.
    """

    def __getinitargs__(self):
        return self.z.copy(), self.α.copy(), self.lB

    def __repr__(self):
        return rf'<{type(self).__name__} z={self.z} α={self.α} lB={self.lB}>'

    def __init__(self, z: float | NDArray, α: float, lB: float=1.):
        self.z = np.atleast_1d(z)
        self.α = np.array(α)
        assert self.α.size == 1
        self.lB = lB

        self.long = GaussianIonLongRange(self)
        self.short = ShortRangeResidual(self)

    def __eq__(self, other):
        if type(other) is not type(self): return False
        return np.allclose(self.z, other.z) and \
               np.all(self.α == other.α) and self.lB == other.lB

    @property
    def nspecies(self):
        return len(self.z)

    def subset(self, species: list | NDArray):
        species = np.atleast_1d(species)
        assert np.all(species >= 0) and np.all(species <= self.nspecies)
        return GaussianIon(self.z[species], self.α, self.lB)

    def potential(self, r: float | NDArray):
        r = np.atleast_1d(r)

        with np.errstate(invalid='ignore'):
            v = self.lB * erf((0.5*self.α)**0.5 * r) / r
        v = np.outer(self.z, self.z)[:,:,None] * v[None,None,:]

        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()
        return v

    def force(self, r: float | NDArray):
        r = np.atleast_1d(r)
        α = self.α

        with np.errstate(invalid='ignore'):
            f = self.lB * (
                    erf((0.5*self.α)**0.5 * r)/r -
                    (2*self.α/np.pi)**0.5 * np.exp(-0.5*self.α*r**2)
                ) / r
        f = np.outer(self.z, self.z)[:,:,None] * f[None,None,:]

        f = np.squeeze(f)
        if f.ndim == 0: f = f.item()
        return f

class ExponentialIonLongRangeOld(Potential):
    r"""Long range part of ExponentialIonOld."""

    def __repr__(self):
        return rf'<{type(self).__name__} z={self.full.z} λ={self.full.λ} lB={self.full.lB}>'

    def __init__(self, full):
        self.full = full

    def __eq__(self, other):
        if type(other) is not type(self): return False
        return self.full == other.full

    @property
    def nspecies(self):
        return self.full.nspecies

    @property
    def α(self):
        """Option for consistency with GaussianIon interface."""
        return self.λ

    @α.setter
    def α(self, value):
        """Option for consistency with GaussianIon interface."""
        self.λ = value

    def potential(self, r: float | NDArray):
        r = np.atleast_1d(r)
        λ = self.full.λ

        with np.errstate(invalid='ignore'):
            v = self.full.lB * (1 - np.exp(-2*r/λ) * (1 + r/λ)) / r
        v = np.outer(self.full.z, self.full.z)[:,:,None] * v[None,None,:]

        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()
        return v

    def potential_fourier(self, k: float | NDArray):
        k = np.atleast_1d(k)

        with np.errstate(invalid='ignore'):
            v = 64*np.pi * self.full.lB / (4 + (k*self.full.λ)**2)**2 / k**2
        v = np.outer(self.full.z, self.full.z)[:,:,None] * v[None,None,:]

        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()
        return v

    def force(self, r: float | NDArray):
        r = np.atleast_1d(r)
        λ = self.full.λ

        with np.errstate(invalid='ignore'):
            f = self.full.lB * (
                    1 - np.exp(-2*r/λ) * (1 + 2*r/λ + 2*(r/λ)**2)
                ) / r**2
        f = np.outer(self.full.z, self.full.z)[:,:,None] * f[None,None,:]

        f = np.squeeze(f)
        if f.ndim == 0: f = f.item()
        return f


class ExponentialIonOld(Potential):
    r"""Interactions between ions with exponentially distributed charges
    (often referred to as "Slater" type in the literature):

        $$\rho_i(r) = \ell_\mathrm{B} z_i
        \frac{e^{-\frac{2 r}{\lambda}}}{\pi \lambda^3}\,,$$

    where $r$ is the distance from the atom centre, $\ell_\mathrm{B}$ is the
    Bjerrum length and $z_i$ is the valence of species $i$.
    """

    def __getinitargs__(self):
        return self.z.copy(), self.λ.copy(), self.lB

    def __repr__(self):
        return rf'<{type(self).__name__} z={self.z} λ={self.λ} lB={self.lB}>'

    def __init__(self, z: float | NDArray, λ: float, lB: float=1.):
        self.z = np.atleast_1d(z)
        self.λ = np.array(λ)
        assert self.λ.size == 1
        self.lB = lB

        self.long = ExponentialIonLongRangeOld(self)
        self.short = ShortRangeResidual(self)

    def __eq__(self, other):
        if type(other) is not type(self): return False
        return np.all(self.z == other.z) and \
               np.all(self.λ == other.λ) and \
               self.lB == other.lB

    @property
    def nspecies(self):
        return len(self.z)

    def subset(self, species: list | NDArray):
        species = np.atleast_1d(species)
        assert np.all(species >= 0) and np.all(species <= self.nspecies)
        return ExponentialIonOld(self.z[species], self.λ, self.lB)

    @property
    def α(self):
        """Option for consistency with GaussianIon interface."""
        return self.λ

    @α.setter
    def α(self, value):
        """Option for consistency with GaussianIon interface."""
        self.λ = value

    def potential(self, r: float | NDArray):
        r = np.atleast_1d(r)
        λ = self.λ

        with np.errstate(invalid='ignore'):
            v = 1 + 11/8*(r/λ) + 3/4*(r/λ)**2 + 1/6*(r/λ)**3
            v = self.lB * (1 - np.exp(-2*r/λ)*v) / r
        v = np.outer(self.z, self.z)[:,:,None] * v[None,None,:]

        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()
        return v

    def force(self, r: float | NDArray):
        r = np.atleast_1d(r)
        λ = self.λ

        with np.errstate(invalid='ignore'):
            f = 1 + 2*r/λ + 2*(r/λ)**2 + 7/6*(r/λ)**3 + 1/3*(r/λ)**4
            f = self.lB * (1 - np.exp(-2*r/λ) * f) / r**2
        f = np.outer(self.z, self.z)[:,:,None] * f[None,None,:]

        f = np.squeeze(f)
        if f.ndim == 0: f = f.item()
        return f

class ExponentialIonLongRange(Potential):
    r"""Long range part of ExponentialIon."""

    def __repr__(self):
        return rf'<{type(self).__name__} z={self.full.z} λ={self.full.λ} lB={self.full.lB}>'

    def __init__(self, full):
        self.full = full

    def __eq__(self, other):
        if type(other) is not type(self): return False
        return self.full == other.full

    @property
    def nspecies(self):
        return self.full.nspecies

    @property
    def α(self):
        """Option for consistency with GaussianIon interface."""
        return self.λ

    @α.setter
    def α(self, value):
        """Option for consistency with GaussianIon interface."""
        self.λ = value

    def potential(self, r: float | NDArray):
        r = np.atleast_1d(r)
        λ = np.atleast_1d(self.full.λ)

        with np.errstate(invalid='ignore'):
            v = np.exp(-2*r/λ[:,None]) * (1 + r/λ[:,None])
            v = 0.5 * (v[None,:,:] + v[:,None,:])
            v = self.full.lB * (1 - v) / r
        v = np.outer(self.full.z, self.full.z)[:,:,None] * v[None,None,:]

        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()
        return v

    def potential_fourier(self, k: float | NDArray):
        k = np.atleast_1d(k)
        λ = np.atleast_1d(self.full.λ)

        with np.errstate(invalid='ignore'):
            Gk = 16 / (4 + λ[:,None]**2 * k**2)**2
            Gk = 0.5 * (Gk[None,:,:] + Gk[:,None,:])
            v = 4 * np.pi * self.full.lB * Gk / k**2
        v = np.outer(self.full.z, self.full.z)[:,:,None] * v[None,None,:]

        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()
        return v

    def force(self, r: float | NDArray):
        r = np.atleast_1d(r)
        λ = self.full.λ

        with np.errstate(invalid='ignore'):
            f = self.full.lB * (
                    1 - np.exp(-2*r/λ) * (1 + 2*r/λ + 2*(r/λ)**2)
                ) / r**2
        f = np.outer(self.full.z, self.full.z)[:,:,None] * f[None,None,:]

        f = np.squeeze(f)
        if f.ndim == 0: f = f.item()
        return f


class ExponentialIon(Potential):
    r"""Interactions between ions with exponentially distributed charges
    (often referred to as "Slater" type in the literature):

        $$\rho_i(r) = \ell_\mathrm{B} z_i
        \frac{e^{-\frac{2 r}{\lambda}}}{\pi \lambda^3}\,,$$

    where $r$ is the distance from the atom centre, $\ell_\mathrm{B}$ is the
    Bjerrum length and $z_i$ is the valence of species $i$.
    """

    def __getinitargs__(self):
        return self.z.copy(), self.λ.copy(), self.lB

    def __repr__(self):
        return rf'<{type(self).__name__} z={self.z} λ={self.λ} lB={self.lB}>'

    def __init__(self, z: float | NDArray, λ: float, lB: float=1.):
        self.z = np.atleast_1d(z)
        self.λ = np.array(λ)
        assert self.λ.size == 1 or self.λ.size == self.z.size
        self.lB = lB

        self.long = ExponentialIonLongRange(self)
        self.short = ShortRangeResidual(self)

    def __eq__(self, other):
        if type(other) is not type(self): return False
        return np.all(self.z == other.z) and \
               np.all(self.λ == other.λ) and \
               self.lB == other.lB

    @property
    def nspecies(self):
        return len(self.z)

    def subset(self, species: list | NDArray):
        species = np.atleast_1d(species)
        assert np.all(species >= 0) and np.all(species <= self.nspecies)
        return ExponentialIon(self.z[species], self.λ, self.lB)

    @property
    def α(self):
        """Option for consistency with GaussianIon interface."""
        return self.λ

    @α.setter
    def α(self, value):
        """Option for consistency with GaussianIon interface."""
        self.λ = value

    def potential(self, r: float | NDArray, rtol: float=1e-8):
        r = np.atleast_1d(r)
        λ = self.λ
        if λ.ndim == 0:
            λ = np.full_like(self.z, λ, dtype=float)

        x = λ**2
        xi, xj = x[:, None, None], x[None, :, None]
        dx = xj - xi
        rr = r[None, None, :]

        def g(x):
            return x**3 * np.exp(-2*rr/x**0.5)
        def gp(x):
            return (3*x**2 + r*x**1.5) * np.exp(-2*rr/x**0.5)
        def gp2(x):
            return (6*x + 4.5*r*x**0.5 + r**2) * np.exp(-2*rr/x**0.5)
        def gp3(x):
            return (6 + 8.25*r/x**0.5 + 4.5*r**2/x + r**3/x**1.5) * np.exp(-2*rr/x**0.5)
        def gp4(x):
            return (1.875*r/x**1.5 + 3.75*r**2/x**2 + 3*r**3/x**2.5 + r**4/x**3) * np.exp(-2*rr/x**0.5)

        gi, gj = g(xi), g(xj)
        gpi, gpj = gp(xi), gp(xj)

        with np.errstate(invalid='ignore', divide='ignore'):
            ψ = 2*(gi - gj) / dx**3 + (gpi + gpj) / dx**2

        # Switch to Taylor expansion to remove numerical instabilities
        # approaching removable singularity as dx->0.
        close = np.abs(dx) < rtol * np.maximum(1, np.abs(xi))
        if np.any(close):
            ψ_lim = gp3(xi)/6 + gp4(xi)/12 * dx
            close = np.broadcast_to(close, ψ.shape)
            ψ[close] = ψ_lim[close]
            
        v = self.lB * (1 - ψ) / rr
        v *= np.outer(self.z, self.z)[:,:,None]

        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()
        return v

    def force(self, r: float | NDArray, rtol: float=1e-8):
        r = np.atleast_1d(r)
        λ = self.λ
        if λ.ndim == 0:
            λ = np.full_like(self.z, λ, dtype=float)

        x = λ**2
        xi, xj = x[:, None, None], x[None, :, None]
        dx = xj - xi
        rr = r[None, None, :]

        def g(x):
            return x**3 * np.exp(-2*rr/x**0.5)
        def gp(x):
            return (3*x**2 + r*x**1.5) * np.exp(-2*rr/x**0.5)
        def gp2(x):
            return (6*x + 4.5*r*x**0.5 + r**2) * np.exp(-2*rr/x**0.5)
        def gp3(x):
            return (6 + 8.25*r/x**0.5 + 4.5*r**2/x + r**3/x**1.5) * np.exp(-2*rr/x**0.5)
        def gp4(x):
            return (1.875*r/x**1.5 + 3.75*r**2/x**2 + 3*r**3/x**2.5 + r**4/x**3) * np.exp(-2*rr/x**0.5)

        # h is derivative of g wrt r.
        def h(x):
            return -2*x**2.5 * np.exp(-2*rr/x**0.5)
        def hp(x):
            return (-5*x**1.5 - 2*r*x) * np.exp(-2*rr/x**0.5)
        def hp2(x):
            return (-7.5*x**0.5 - 7*r - 2*r**2/x**0.5) * np.exp(-2*rr/x**0.5)
        def hp3(x):
            return (-3.75/x**0.5 - 7.5*r/x - 6*r**2/x**1.5 - 2*r**3/x**2) * np.exp(-2*rr/x**0.5)
        def hp4(x):
            return (1.875/x**1.5 + 3.75*r/x**2 + 1.5*r**2/x**2.5 - 2*r**3/x**3 - 2*r**4/x**3.5) * np.exp(-2*rr/x**0.5)

        gi, gj = g(xi), g(xj)
        gpi, gpj = gp(xi), gp(xj)
        hi, hj = h(xi), h(xj)
        hpi, hpj = hp(xi), hp(xj)

        with np.errstate(invalid='ignore', divide='ignore'):
            ψ = 2*(gi - gj) / dx**3 + (gpi + gpj) / dx**2
            ψp = 2*(hi - hj) / dx**3 + (hpi + hpj) / dx**2

        # Switch to Taylor expansion to remove numerical instabilities
        # approaching removable singularity as dx->0.
        close = np.abs(dx) < rtol * np.maximum(1, np.abs(xi))
        if np.any(close):
            ψ_lim = gp3(xi)/6 + gp4(xi)/12 * dx
            ψp_lim = hp3(xi)/6 + hp4(xi)/12 * dx
            close = np.broadcast_to(close, ψ.shape)
            ψ[close] = ψ_lim[close]
            ψp[close] = ψp_lim[close]
            
        f = self.lB * (1 - ψ) / rr**2 + self.lB * ψp / rr
        f *= np.outer(self.z, self.z)[:,:,None]

        f = np.squeeze(f)
        if f.ndim == 0: f = f.item()
        return f

import pytest
@pytest.mark.parametrize('l', [1, 2])
def test_consistency(l):
    v1 = ExponentialIonOld([1, -1], l)
    v2 = ExponentialIon([1, -1], l)
    v3 = ExponentialIon([1, -1], [l, l])
    v4 = ExponentialIon([1, -1], [l, 0.5*l])
    r = np.linspace(0, 1, 4)[1:]
    a = v1.long.potential_fourier(r)
    b = v2.long.potential_fourier(r)
    c = v3.long.potential_fourier(r)
    d = v4.long.potential_fourier(r)
    assert np.allclose(a, b)
    assert np.allclose(b, c)
    assert not np.allclose(c, d)
    a = v1.long.potential(r)
    b = v2.long.potential(r)
    c = v3.long.potential(r)
    d = v4.long.potential(r)
    assert np.allclose(a, b)
    assert np.allclose(b, c)
    assert not np.allclose(c, d)
    a = v1.potential(r)
    b = v2.potential(r)
    c = v3.potential(r)
    d = v4.potential(r)
    assert np.allclose(a, b)
    assert np.allclose(b, c)
    assert not np.allclose(c, d)
    a = v1.force(r)
    b = v2.force(r)
    c = v3.force(r)
    d = v4.force(r)
    assert np.allclose(a, b)
    assert np.allclose(b, c)
    assert not np.allclose(c, d)

import pytest
@pytest.mark.parametrize('cls', [GaussianIon, ExponentialIon])
def test_ion(cls):
    import pickle
    def test_copy(v, v2):
        assert v2 is not v
        assert v2 == v
        assert np.all(v.α == v2.α)
        assert np.all(v.z == v2.z)
        assert v.lB == v2.lB
        v2.α += 1
        v2.z *= 2
        v2.lB += 1
        assert v2 != v
        assert np.all(v.α != v2.α)
        assert np.all(v.z != v2.z)
        assert v.lB != v2.lB

    v = cls([1, -1], 1)
    assert np.allclose(v.copy().potential(1.), v.potential(1.))
    test_copy(v, v.copy())
    test_copy(v, pickle.loads(pickle.dumps(v)))

    from scipy.optimize import approx_fprime
    r = np.linspace(0, 10, 100)[1:]

    for i in range(v.nspecies):
        for j in range(v.nspecies):
            for vv in [v.long, v, v.short]:
                f = lambda r: vv.potential(r)[i,j]
                exact = np.array([-approx_fprime(rr, f) for rr in r]).reshape(-1)
                force = vv.force(r)[i,j]
                assert np.allclose(vv.force(r)[i,j], exact, rtol=1e-5)

    short, long = v.short.potential(r), v.long.potential(r)
    assert np.allclose(v.potential(r), short + long)


class LennardJones(Potential):
    r"""Standard truncated Lennard-Jones potential used in atomic simulations:

        $$v(r) = 4\epsilon \left( \left( \frac{\sigma}{r} \right)^12 - \left( \frac{\sigma}{r} \right)^6 \right)\,.$$

    If truncated at some rcut < infinity) then this becomes

        $$v_\text{truncate}(r) = v(r) - v(rcut)\,.$$
    """

    def __getinitargs__(self):
        return self.sigma, self.epsilon, self.rcut

    def __init__(self, sigma: float = 1.,
                 epsilon: float = 1.,
                 rcut: float = None):
        self.sigma = sigma
        self.epsilon = epsilon
        if rcut is None: rcut = 2.5*self.sigma
        self.rcut = rcut

        r6inv = (self.sigma/self.rcut)**6
        self.vshift = 4*self.epsilon * (r6inv**2 - r6inv)

    def __call__(self, *args, **kwargs):
        return self.potential(*args, **kwargs)

    def __eq__(self, other):
        if type(other) is not type(self): return False
        return np.all(self.sigma == other.sigma) and \
               np.all(self.epsilon == other.epsilon) and \
               np.all(self.rcut == other.rcut) and \
               self.vshift == other.vshift

    @property
    def nspecies(self):
        return 1

    def subset(self, species: list | NDArray):
        species = np.atleast_1d(species)
        assert np.all(species >= 0) and np.all(species <= self.nspecies)
        indices = np.ix_(species, species)
        return LennardJones(self.sigma[indices], self.epsilon[indices],
                            self.rcut[indices])

    def potential(self, r: float | NDArray):
        r = np.atleast_1d(r)

        r6inv = (self.sigma/r)**6
        v = 4*self.epsilon * (r6inv**2 - r6inv) - self.vshift
        v[r >= self.rcut] = 0.

        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()
        return v

    def force(self, r: float | NDArray):
        r = np.atleast_1d(r)

        rinv = self.sigma/r
        r6inv = rinv**6
        f = 4*self.epsilon * (12*r6inv**2 - 6*r6inv) / r
        f[r >= self.rcut] = 0.

        f = np.squeeze(f)
        if f.ndim == 0: f = f.item()
        return f


def test_lj():
    import pickle
    def test_copy(v, v2):
        assert v2 is not v
        assert v == v2
        assert np.all(v.sigma == v2.sigma)
        assert np.all(v.epsilon == v2.epsilon)
        assert np.all(v.rcut == v2.rcut)
        v2.sigma += 1
        v2.epsilon += 1
        v2.rcut += 1
        assert v != v2
        assert np.all(v.sigma != v2.sigma)
        assert np.all(v.epsilon != v2.epsilon)
        assert np.all(v.rcut != v2.rcut)

    r = np.linspace(1, 10, 100)

    # Tests for single-component systems.

    v = LennardJones()
    assert np.isscalar(v.potential(1.))
    assert not np.isscalar(v.potential(r))
    assert v.copy().potential(1.) == v.potential(1.)
    test_copy(v, v.copy())
    test_copy(v, pickle.loads(pickle.dumps(v)))

    from scipy.optimize import approx_fprime
    exact = np.array([-approx_fprime(rr, v.potential) for rr in r]).reshape(-1)
    assert np.allclose(v.force(r), exact, rtol=1e-6)

    rcut = 2.
    v = LennardJones(rcut=rcut)
    phi = v.potential(r)
    assert np.allclose(phi[r > rcut], 0.)
    assert np.all(~np.isclose(phi[r < rcut], 0.))


class Gaussian(Potential):
    r"""A simple Gaussian potential. This is primarily used to test Ng
    splitting (cf. `GaussianSplit` and `pyhnc.OrnsteinZernikeSolver`)."""

    def __getinitargs__(self):
        return (self.alpha.copy(),)

    def __repr__(self):
        return rf'<Gaussian α={self.alpha.tolist()}>'

    def __init__(self, alpha: float | NDArray=1.):
        alpha = np.atleast_2d(alpha)
        assert alpha.shape[0] == alpha.shape[1]
        self.alpha = alpha

    def __eq__(self, other):
        if type(other) is not type(self): return False
        return np.all(self.alpha == other.alpha)

    @property
    def nspecies(self):
        return len(self.alpha)

    def subset(self, species: list | NDArray):
        species = np.atleast_1d(species)
        assert np.all(species >= 0) and np.all(species <= self.nspecies)
        indices = np.ix_(species, species)
        return Gaussian(self.alpha[indices])

    def potential(self, r: float | NDArray):
        r = np.atleast_1d(r)

        α = self.alpha[:,:,None]
        v = (α/np.pi)**1.5 * np.exp(-α * r**2)

        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()
        return v

    def potential_fourier(self, k: float | NDArray):
        k = np.atleast_1d(k)

        α = self.alpha[:,:,None]
        v = np.exp(-k**2 / (4*α))

        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()
        return v

    def force(self, r: float | NDArray):
        r = np.atleast_1d(r)

        α = self.alpha[:,:,None]
        f = 2*r * (α/np.pi)**1.5 * np.exp(-α * r**2)

        f = np.squeeze(f)
        if f.ndim == 0: f = f.item()
        return f

class GaussianSplit(Gaussian):
    r"""Helper class so that whole Gaussian will be splintered off while
    solving the Ornstein-Zernike equation (cf. `pyhnc.OrnsteinZernikeSolver)
    to test Ng splitting."""

    def __init__(self, alpha: float | NDArray=1.):
        super().__init__(alpha)
        self.long = Gaussian(alpha)
        self.short = ShortRangeResidual(self)


def test_gaussian():
    import pickle
    def test_copy(v, v2):
        assert v2 is not v
        assert v2 == v
        assert np.all(v.alpha == v2.alpha)
        v2.alpha += 1
        assert v2 != v
        assert np.all(v.alpha != v2.alpha)

    r = np.linspace(1, 10, 100)

    # Tests for single-component systems.

    v = Gaussian()
    assert np.isscalar(v.potential(1.))
    assert not np.isscalar(v.potential(r))
    assert v.copy().potential(1.) == v.potential(1.)
    test_copy(v, v.copy())
    test_copy(v, pickle.loads(pickle.dumps(v)))

    from scipy.optimize import approx_fprime
    exact = np.array([-approx_fprime(rr, v.potential) for rr in r]).reshape(-1)
    assert np.allclose(v.force(r), exact, rtol=1e-6)

class DPDGaussianIon(Potential):
    r"""DPD bead with an additional Gaussian-distributed soft electrostatic
    potential. Cf. documentation for `DPD` and `GaussianIon` for details on
    these components.
    """

    def __getinitargs__(self):
        return self.dpd.A.copy(), self.ion.z.copy(), \
               self.ion.α.copy(), self.dpd.rcut.copy(), self.ion.lB

    def __repr__(self):
        return rf'<DPDGaussianIon dpd={self.dpd} ion={self.ion}>'

    def __init__(self, A: float | NDArray,
                 z: float | NDArray, α: float,
                 rcut: float | NDArray=1.,
                 lB: float=1.):

        self.dpd = DPD(A, rcut)
        self.ion = GaussianIon(z, α, lB)
        assert self.dpd.nspecies == self.ion.nspecies
        self.long = self.ion.long
        self.short = ShortRangeResidual(self)

    def __eq__(self, other):
        if type(other) is not type(self): return False
        return self.dpd == other.dpd and self.ion == other.ion

    @property
    def nspecies(self):
        assert self.dpd.nspecies == self.ion.nspecies
        return self.dpd.nspecies

    def subset(self, species: list | NDArray):
        species = np.atleast_1d(species)
        assert np.all(species >= 0) and np.all(species <= self.nspecies)
        indices = np.ix_(species, species)
        return DPDGaussianIon(self.dpd.A[indices],
                              self.ion.z[species],
                              self.ion.α,
                              self.dpd.rcut[indices],
                              self.ion.lB)

    def potential(self, r: float | NDArray):
        return self.dpd.potential(r) + self.ion.potential(r)

    def force(self, r: float | NDArray):
        return self.dpd.force(r) + self.ion.force(r)


class DPDExponentialIon(Potential):
    r"""DPD bead with an additional Exponential-distributed soft electrostatic
    potential. Cf. documentation for `DPD` and `ExponentialIon` for details on
    these components.
    """

    def __getinitargs__(self):
        return self.dpd.A.copy(), self.ion.z.copy(), \
               self.ion.λ.copy(), self.dpd.rcut.copy(), self.ion.lB

    def __repr__(self):
        return rf'<DPDExponentialIon dpd={self.dpd} ion={self.ion}>'

    def __init__(self, A: float | NDArray,
                 z: float | NDArray, λ: float,
                 rcut: float | NDArray=1.,
                 lB: float=1.):

        self.dpd = DPD(A, rcut)
        self.ion = ExponentialIon(z, λ, lB)
        assert self.dpd.nspecies == self.ion.nspecies
        self.long = self.ion.long
        self.short = ShortRangeResidual(self)

    def __eq__(self, other):
        if type(other) is not type(self): return False
        return self.dpd == other.dpd and self.ion == other.ion

    @property
    def nspecies(self):
        assert self.dpd.nspecies == self.ion.nspecies
        return self.dpd.nspecies

    def subset(self, species: list | NDArray):
        species = np.atleast_1d(species)
        assert np.all(species >= 0) and np.all(species <= self.nspecies)
        indices = np.ix_(species, species)
        return DPDExponentialIon(self.dpd.A[indices],
                                 self.ion.z[species],
                                 self.ion.λ,
                                 self.dpd.rcut[indices],
                                 self.ion.lB)

    def potential(self, r: float | NDArray):
        return self.dpd.potential(r) + self.ion.potential(r)

    def force(self, r: float | NDArray):
        return self.dpd.force(r) + self.ion.force(r)


if __name__ == '__main__':
    test_dpd()
    test_consistency(1)
    test_consistency(2)
    for ion in GaussianIon, ExponentialIon: test_ion(ion)
    test_lj()
    test_gaussian()
