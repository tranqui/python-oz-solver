#!/usr/bin/env python3

import argparse
import numpy as np
from numpy import pi as π
from numpy import exp, sqrt

# Standard Pitzer model, revised
# Values of β0, β1, Cϕ from Table 1 in J. Chem. Eng. Data 33, 177-84 (1988)

pitzer_coeffs = {
    "NaCl": (0.07722, 0.25183, 0.00106),
    "KCl":  (0.04661, 0.22341,-0.00044),
    "LiI":  (0.14661, 0.75394, 0.02126) }

# Modified Pitzer model
# Values of b, β0, CMX from Table 2 in Ind. Eng. Chem. Res. 41(5), 1031-7 (2002)

modified_pitzer_coeffs = {
    "NaCl": (2.22718, 0.05383, 0.00134),
    "KCl":  (1.99609, 0.02699, 0.00043),
    "LiI":  (3.45328, 0.15671, 0.00495) }

T = 298 # temperature
ρw, εr = 0.997e3, 78.4 # water properties at 298 K
NA, kB, e, ε0 = 6.022e23, 1.38e-23, 1.602e-19, 8.854e-12 # fundamental constants

lB = e**2 / (4*π*εr*ε0*kB*T) # Bjerrum length in water

Aϕ = 1/3 * sqrt(2*π*NA*ρw) * lB**(3/2) # Debye-Hückel coefficient
print(f'Aϕ (calculated) = {Aϕ}')

Aϕ = 0.3915 # to match above Ind. Eng. Chem. Res. 41(5), 1031-7 (2002)
print(f'Aϕ (assumed) = {Aϕ}')

print(f'Πref / molality (1:1 electrolyte) = {2*NA*kB*T*ρw*1e-5} × 10^5 Pa.kg/m³')
print(f'Πref / (∑i mi) = RT ρw = {NA*kB*T*ρw*1e-5} × 10^5 Pa.kg/m³')

parser = argparse.ArgumentParser()
parser.add_argument('--name', default=None, help='the name of the electrolyte')
parser.add_argument('--modified', action='store_true', help='use modified Pitzer equation')
args = parser.parse_args()

if not args.name or not args.name in pitzer_coeffs:
    print("use with --name=<electrolyte>, choice of ",
          ", ".join(pitzer_coeffs.keys()))
    exit(0)

# For 1:1 electrolytes the stoichiometry coefficients and valencies are all unity

m = np.insert(np.logspace(-3, np.log10(10), 53), 0, 0.0) # prepend 0.0 to logspace

rootI = sqrt(m) # sqrt of ionic strength

if args.modified:

    b, β0, CMX = modified_pitzer_coeffs[args.name]
    Bϕ = β0
    Cϕ = 2 * CMX
    
else:
        
    α, b = 2, 1.2
    β0, β1, Cϕ = pitzer_coeffs[args.name]
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

result_file = args.name + '_pitzer' + ('_modified.dat' if args.modified else '.dat')
np.savetxt(result_file, result, delimiter='\t')
print("Written (m, Πref, Π, Φ) to", result_file)
