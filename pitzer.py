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
    'NaCl':  ( 0.07722,  0.25183,  0.00106),
    'NaNO3': ( 0.00388,  0.21151, -0.00006),
    'NaI':   ( 0.13463,  0.19479, -0.00117),
    'NaBr':  ( 0.11077,  0.13760, -0.00153),
    'KCl':   ( 0.04661,  0.22341, -0.00044),
    'KNO3':  (-0.08511,  0.10518,  0.00773),
    'KI':    ( 0.07253,  0.27710, -0.00381),
    'KBr':   ( 0.05592,  0.22094, -0.00162),
    'LiCl':  ( 0.20972, -0.34380, -0.00433),
    'LiNO3': ( 0.13008,  0.04957, -0.00382),
    'LiI':   ( 0.14661,  0.75394,  0.02126),
    'LiBr':  ( 0.24554, -0.44244, -0.00293),
    'CsCl':  ( 0.03643, -0.01169, -0.00096),
    'CsNO3': (-0.13004,  0.08169,  0.03018),
    'CsI':   ( 0.02121,  0.07307, -0.00307),
    'CsBr':  ( 0.02311,  0.04587,  0.00092)
    }

# Modified Pitzer model
# Values of b, β0, CMX from Table 2 in Ind. Eng. Chem. Res. 41(5), 1031-7 (2002)

modified_pitzer_coeffs = {
    'NaCl':  (2.22718,  0.05383,  0.00134),
    'NaNO3': (2.25248, -0.04164,  0.00335),
    'NaI':   (2.14921,  0.11339,  0.00000),
    'NaBr':  (2.06796,  0.08821,  0.00000),
    'KCl':   (1.99609,  0.02699,  0.00043),
    'KNO3':  (1.45780, -0.09111,  0.00379),
    'KI':    (2.30629,  0.04747, -0.00101),
    'KBr':   (2.06975,  0.03407,  0.00000),
    'LiCl':  (0.41585,  0.27034, -0.00340),
    'LiNO3': (2.33706,  0.11340, -0.00164),
    'LiI':   (3.45328,  0.15671,  0.00495),
    'LiBr':  (0.10632,  0.37681, -0.00402),
    'CsCl':  (1.34596,  0.02923, -0.00019),
    'CsNO3': (1.60904, -0.15003,  0.01718),
    'CsI':   (1.42147,  0.01336, -0.00126),
    'CsBr':  (1.47723,  0.01030,  0.00118)
    }

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
