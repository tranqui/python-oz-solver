#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi as π
from numpy import exp, sqrt

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

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

for ion in ions:
    mol = Chem.MolFromSmiles(f'[{ion}]')
    print(f'{ion:>4}: {ExactMolWt(mol):>8.4f} g/mol')

print()

T = 298 # temperature
ρw, εr = 0.997e3, 78.4 # water properties at 298 K
NA, kB, e, ε0 = 6.022e23, 1.38e-23, 1.602e-19, 8.854e-12 # fundamental constants

lB = e**2 / (4*π*εr*ε0*kB*T) # Bjerrum length in water

Aϕ = 1/3 * sqrt(2*π*NA*ρw) * lB**(3/2) # Debye-Hückel coefficient
print(f'Aϕ (calculated) = {Aϕ}')

# Aϕ = 0.3915 # to match above Ind. Eng. Chem. Res. 41(5), 1031-7 (2002)
# print(f'Aϕ (assumed) = {Aϕ}')

print(f'Πref / molality (1:1 electrolyte) = {2*NA*kB*T*ρw*1e-5} × 10^5 Pa.kg/m³')
print(f'Πref / (∑i mi) = RT ρw = {NA*kB*T*ρw*1e-5} × 10^5 Pa.kg/m³')

# For 1:1 electrolytes the stoichiometry coefficients and valencies are all unity

m = np.insert(np.geomspace(1e-3, 10, 51), 0, 0.0) # prepend 0.0 to logspace

rootI = sqrt(m) # sqrt of ionic strength

for name in pitzer_coeffs:
    for modified in [False, True]:

        if modified:

            b, β0, CMX = modified_pitzer_coeffs[name]
            Bϕ = β0
            Cϕ = 2 * CMX

        else:

            α, b = 2, 1.2
            β0, β1, Cϕ = pitzer_coeffs[name]
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

        molar_mass_water = 18.01528e-3 # kg/mol

        x = m / (m + 1/molar_mass_water)
        x = m
        y = Π
        # y = Φ

        if modified:
            plt.plot(x, y, ':', c=pl.get_color())
        else:
            label = name.replace('+', '').replace('-', '')
            label = rf'$\ce{{{label}}}$'
            pl, = plt.plot(x, y, label=label)

plt.legend(loc='best')
plt.ylabel('osmotic pressure $\Pi$ / bar')
# plt.ylabel('osmotic coefficient $\phi$')
# plt.xlabel(r'mole fraction $x_z \equiv x_+ + x_-$')
plt.xlabel(r'molality $m$ / $\si{\mol\per\kilogram}$')

plt.ylim([0, 500])
plt.xlim([0, 10])

plt.show()
