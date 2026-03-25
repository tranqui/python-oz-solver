#!/usr/bin/env python3

import numpy as np
from scipy.optimize import root_scalar
from functools import cached_property

class BarostatSolver:

    def __init__(self, reference_system, potential, solver=None):
        self.reference_system = reference_system.copy()
        self.solver = reference_system.copy() if solver is None else solver.copy()
        self.pressure = self.reference_system.pressure

        self.potential = potential
        self.first = solver is None
        self.solutions = []

    def residual(self, xi, rho, **kwargs):
        ρi = xi*rho
        assert np.isclose(np.sum(ρi), rho)
        self.solver.solve(self.potential, ρi, **kwargs)
        assert self.solver.converged
        self.first = False

        return self.solver.pressure - self.pressure

    @property
    def composition(self):
        xi = np.empty((len(self.solutions), self.potential.nspecies))
        for j, sol in enumerate(self.solutions):
            xi[j] = sol.rho / np.sum(sol.rho)
        return xi

    @property
    def number_density(self):
        ni = np.empty((len(self.solutions), self.potential.nspecies))
        for j, sol in enumerate(self.solutions):
            ni[j] = sol.rho
        return ni

    @property
    def density(self):
        return self.number_density

    @cached_property
    def osmotic_pressure(self):
        return np.array([sol.osmotic_pressure for sol in self.solutions])

    @cached_property
    def chemical_potential(self):
        return np.array([sol.excess_chemical_potential[0] + np.log(sol.density[0]) for sol in self.barostat.solutions])

    def solve(self, xi, eps=1e-2, ρ0=None, **kwargs):
        xi = np.atleast_1d(xi)
        assert np.isclose(np.sum(xi), 1)
        assert np.all(xi >= 0)

        if ρ0 is None:
            if self.first: ρ0 = np.sum(self.reference_system.density)
            else: ρ0 = np.sum(self.solver.density)
        optim = root_scalar(lambda p: self.residual(xi, p, **kwargs),
                            bracket=(ρ0-eps, ρ0+eps))
        ρi = xi*optim.root
        sol = self.solver.copy().solve(self.potential, ρi, **kwargs)
        self.solutions += [sol]
