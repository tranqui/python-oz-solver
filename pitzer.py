#!/usr/bin/env python3

from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi as π
from numpy import exp, sqrt
from scipy.constants import N_A as NA, Boltzmann as kB, e, epsilon_0 as ε0
from scipy.constants import atmosphere as atmospheric_pressure

import chemicals
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt

# Water constants

water = Chem.MolFromSmiles('O')
water_molar_mass = 1e-3 * MolWt(water) # [kg/mol]


# Standard Pitzer model, revised
# Values of β0, β1, Cϕ from Table 1 in J. Chem. Eng. Data 33, 177-84 (1988)

pitzer_coeffs = {
    'Na+Cl-': (0.07722, 0.25183, 0.00106),
    'K+Cl-':  (0.04661, 0.22341,-0.00044),
    'Li+I-':  (0.14661, 0.75394, 0.02126) }

# Modified Pitzer model
# Values of b, β0, CMX from Table 2 in Ind. Eng. Chem. Res. 41(5), 1031-7 (2002)

modified_pitzer_coeffs = {
    'Na+Cl-': (2.22718, 0.05383, 0.00134),
    'K+Cl-':  (1.99609, 0.02699, 0.00043),
    'Li+I-':  (3.45328, 0.15671, 0.00495) }

ions = []
for name in pitzer_coeffs:
    import re
    split = [s for s in re.split(r'(?=[A-Z])', name) if s != '']
    for s in split:
        if s not in ions: ions += [s]


class PitzerModel:
    def __init__(self, name):
        self.regular_coeffs = pitzer_coeffs[name]
        self.modified_coeffs = modified_pitzer_coeffs[name]

    def __call__(self, T, m, modified=False,
                 pressure=atmospheric_pressure):

        ρw = chemicals.iapws.iapws97_rho(T, pressure)
        εr = chemicals.permittivity.permittivity_IAPWS(T, ρw)
        lB = e**2 / (4*π*εr*ε0*kB*T) # Bjerrum length in water

         # Debye-Hückel coefficient
        Aϕ = 1/3 * sqrt(2*π*NA*ρw) * lB**(3/2)
        # Aϕ = 0.3915 # to match above Ind. Eng. Chem. Res. 41(5), 1031-7 (2002)

        rootI = sqrt(m)
        if modified:
            b, β0, CMX = self.modified_coeffs
            Bϕ = β0
            Cϕ = 2 * CMX
        else:
            β0, β1, Cϕ = self.regular_coeffs
            α, b = 2, 1.2
            Bϕ = β0 + β1 * exp(-α*rootI)

        Φ = 1 - Aϕ*rootI/(1 + b*rootI) + Bϕ*m + Cϕ*m**2

        # Note definitions in Guggenheim: Π ⟨V1⟩ / RT = Φ ∑i ri where ri = ni/n1 = M1 × mi
        # are the solute-solvent mole ratios, with the 'standard' reference M1 being
        # the molar mass of solvent ('1' being the solvent).  Hence Π = Φ RT M1/⟨V1⟩ ∑i mi
        # = Φ RT ρ1 ∑i mi where ρ1 = M1/⟨V1⟩ ≈ ρw is the (average) solvent mass density.
        # The result here is expressed in bar = 10^5 Pa.

        Πref = 1e-5 * NA*kB*T*ρw * 2*m

        Π = Πref * Φ

        result = np.vstack((m, Πref, Π, Φ)).transpose()

        x = m / (m + 1/water_molar_mass) # mole fraction

        return SimpleNamespace(m=m, Φ=Φ, Πref=Πref, Π=Π, x=x)


if __name__ == '__main__':

    for ion in ions:
        mol = Chem.MolFromSmiles(f'[{ion}]')
        print(f'{ion:>4}: {MolWt(mol):>8.4f} g/mol')

    T = 298 # temperature
    m = np.insert(np.geomspace(1e-3, 10, 51), 0, 0.0) # prepend 0.0 to logspace

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(3.375, 3.375))

    for name in pitzer_coeffs:
        model = PitzerModel(name)

        for modified in [False, True]:
            result = model(T, m, modified)

            if modified:
                ax1.plot(result.x, result.Φ, ':', c=pl.get_color())
                ax2.plot(result.m, result.Π, ':', c=pl.get_color())
            else:
                label = name.replace('+', '').replace('-', '')
                label = rf'$\ce{{{label}}}$'
                pl, = ax1.plot(result.x, result.Φ, label=label)
                ax2.plot(result.m, result.Π)


    ax1.set_xlabel(r'mole fraction $x_z \equiv x_+ + x_-$')
    ax1.set_ylabel(r'osmotic\\coefficient $\phi$')
    ax1.legend(loc='best')
    ax2.set_xlabel(r'molality $m$ / $\si{\mol\per\kilogram}$')
    ax2.set_ylabel(r'osmotic\\pressure $\Pi$ / bar')

    plt.show()
