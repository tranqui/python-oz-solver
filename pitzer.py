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

# Specify ions in dataset

cations = ['Na+', 'Li+', 'K+', 'Rb+', 'Cs+', 'NH4+', 'Ca+2', 'Mg+2']
anions = ['Cl-', 'NO3-', 'I-', 'Br-', 'SO4-2']
ions = cations + anions

# Conversion to different formats

name_to_smiles = {}
for name in cations + anions:
    name_to_smiles[name] = f'[{name}]'
name_to_smiles['O'] = 'O'
name_to_smiles['NO3-'] = '[N+](=O)([O-])[O-]'
name_to_smiles['SO4-2'] = 'S(=O)(=O)([O-])[O-]'

def ce(s):
    """Convert to LaTeX mhchem format."""
    if '+' in s or '-' in s:
        if s[-1] not in '+-':
            if '+' in s:
                a, b = s.split('+')
                b = b + '+'
            else:
                a, b = s.split('-')
                b = b + '-'
            s = a + rf'^{{{b}}}'
    return rf'\ce{{{s}}}'


# Standard Pitzer model, revised
# Values of β0, β1, Cϕ, mmax from TableS 1-3 in:
# J. Chem. Eng. Data 33, 177-84 (1988)

pitzer_coeffs = {
    # Max molality given on end of line
    'NaCl'  : ( 0.07722,  0.25183,  0.00106,  6.144),
    'NaNO3' : ( 0.00388,  0.21151, -0.00006, 10.830),
    'NaI'   : ( 0.13463,  0.19479, -0.00117, 12.000),
    'NaBr'  : ( 0.11077,  0.13760, -0.00153,  9.000),
    'NaSO4' : ( 0.04604,  0.93350, -0.00483,  1.750),

    'KCl'   : ( 0.04661,  0.22341, -0.00044,  4.803),
    'KNO3'  : (-0.08511,  0.10518,  0.00773,  3.500),
    'KI'    : ( 0.07253,  0.27710, -0.00381,  4.500),
    'KBr'   : ( 0.05592,  0.22094, -0.00162,  5.500),
    'KSO4'  : ( 0.07548,  0.44371,  0.00000,  0.692),

    'LiCl'  : ( 0.20972, -0.34380, -0.00433, 19.219),
    'LiNO3' : ( 0.13008,  0.04957, -0.00382, 20.000),
    'LiI'   : ( 0.14661,  0.75394,  0.02126,  3.000),
    'LiBr'  : ( 0.24554, -0.44244, -0.00293, 20.000),
    'LiSO4' : ( 0.14473,  1.29952, -0.00616,  3.000),

    'RbCl'  : ( 0.04660,  0.12983, -0.00163,  7.800),
    'RbNO3' : (-0.08174, -0.03175,  0.00624,  4.500),
    'RbI'   : ( 0.03902,  0.15224, -0.00095,  5.000),
    'RbBr'  : ( 0.03868,  0.16723, -0.00123,  5.000),
    'RbSO4' : ( 0.09123,  0.77863, -0.01282,  0.070),

    'CsCl'  : ( 0.03643, -0.01169, -0.00096, 11.000),
    'CsNO3' : (-0.13004,  0.08169,  0.03018,  1.500),
    'CsI'   : ( 0.02121,  0.07307, -0.00307,  3.000),
    'CsBr'  : ( 0.02311,  0.04587,  0.00092,  5.000),
    'CsSO4' : ( 0.14174,  0.69456, -0.02686,  1.831),

    'NH4Cl' : ( 0.05191,  0.17937, -0.00301,  7.405),
    'NH4NO3': (-0.01476,  0.13826,  0.00029, 25.954),
    'NH4I'  : ( 0.05701,  0.31566, -0.00308,  7.500),
    'NH4SO4': ( 0.04841,  1.13240, -0.00155,  5.500),

    'CaCl'  : ( 0.32579,  1.38412, -0.00174,  6.000),
    'CaNO3' : ( 0.17030,  2.02106, -0.00690,  6.000),
    'CaI'   : ( 0.43225,  1.84879,  0.00085,  1.915),
    'CaBr'  : ( 0.33899,  2.04551,  0.01067,  6.000),
    'CaSO4' : ( 0.20000,  3.77620,  0      ,  0.020),

    'MgCl'  : ( 0.35573,  1.61738,  0.00474,  5.750),
    'MgNO3' : ( 0.34284,  2.68244, -0.00723,  5.000),
    'MgI'   : ( 0.49161,  1.78273,  0.00780,  5.000),
    'MgBr'  : ( 0.43460,  1.73184,  0.00275,  5.610),
    'MgSO4' : ( 0.22438,  3.3067 , -40.493, 0.02512, 3),
    }

# Modified Pitzer model
# Values of b, β0, CMX, mmax from Tables 2 & 3 in:
# [1] Ind. Eng. Chem. Res. 41(5), 1031-7 (2002)
# And Table 2 in:
# [2] Ind. Eng. Chem. Res. 46(19) (2007)

modified_pitzer_coeffs = {
    'NaCl' : (2.22718,  0.05383,  0.00134,  6.14),
    'NaNO3': (2.25248, -0.04164,  0.00335, 10.83),
    'NaI'  : (2.14921,  0.11339,  0.00000, 12),
    'NaBr' : (2.06796,  0.08821,  0.00000,  9),
    'NaSO4': (1.91900, -0.04330,  0.00389,  4.445),

    'KCl'  : (1.99609,  0.02699,  0.00043, 5),
    'KNO3' : (1.45780, -0.09111,  0.00379, 3.5),
    'KI'   : (2.30629,  0.04747, -0.00101, 4.5),
    'KBr'  : (2.06975,  0.03407,  0.00000, 5.5),
    'KSO4' : (1.36345,  0.10562, -0.01711, 0.692),

    'LiCl' : (0.41585,  0.27034, -0.00340, 19.22),
    'LiNO3': (2.33706,  0.11340, -0.00164, 20),
    'LiI'  : (3.45328,  0.15671,  0.00495,  3),
    'LiBr' : (0.10632,  0.37681, -0.00402, 20),
    'LiSO4': (2.09807,  0.06128,  0.00119,  3.165),

    'RbSO4': (1.77607,  0.01603,  0.00000, 1.707),

    'CsCl' : (1.34596,  0.02923, -0.00019, 11),
    'CsNO3': (1.60904, -0.15003,  0.01718,  1.5),
    'CsI'  : (1.42147,  0.01336, -0.00126,  3),
    'CsBr' : (1.47723,  0.01030,  0.00118,  5),
    'CsSO4': (1.66820,  0.08523, -0.00714,  1.631),

    'NH4Cl': (1.94663,  0.04189, -0.00079, 7.41),

    'CaCl' : (1.10809,  0.44947, -0.00608, 10),
    'CaNO3': (2.45544,  0.08678, -0.00022,  7.785),
    'CaI'  : (2.99420,  0.30217,  0.00649,  2),
    'CaBr' : (2.73540,  0.25292,  0.00587,  6),

    'MgCl' : (2.37916,  0.28486,  0.00331,  5), # from [2]
    'MgNO3': (2.64818,  0.23651,  0.00052,  5),
    'MgI'  : (2.51096,  0.42172,  0.00434,  5), # from [2]
    'MgBr' : (2.44928,  0.35950,  0.00293,  5), # from [2]
    }

class PitzerModel:
    def __init__(self, cation, anion):
        self.cation = cation
        self.anion = anion

        mol1 = Chem.MolFromSmiles(name_to_smiles[cation])
        mol2 = Chem.MolFromSmiles(name_to_smiles[anion])
        self.zM, self.zX = [Chem.GetFormalCharge(mol) for mol in [mol1, mol2]]
        assert self.zM >= 1
        assert self.zX <= -1
        d = np.gcd(self.zM, self.zX)
        self.vM, self.vX = -self.zX//d, self.zM//d    # number of each ion
        assert self.vM*self.zM + self.vX*self.zX == 0 # charge neutrality

        try: self.regular_coeffs = pitzer_coeffs[self.name]
        except KeyError: self.modified_coeffs = None
        try: self.modified_coeffs = modified_pitzer_coeffs[self.name]
        except KeyError: self.modified_coeffs = None

    @property
    def name(self):
        return f'{cation.split('+')[0]}{anion.split('-')[0]}'

    def __call__(self, T, m, modified=False,
                 pressure=atmospheric_pressure):

        ρw = chemicals.iapws.iapws97_rho(T, pressure)
        εr = chemicals.permittivity.permittivity_IAPWS(T, ρw)
        lB = e**2 / (4*π*εr*ε0*kB*T) # Bjerrum length in water

         # Debye-Hückel coefficient
        Aϕ = 1/3 * sqrt(2*π*NA*ρw) * lB**(3/2)
        # Aϕ = 0.3915 # to match above Ind. Eng. Chem. Res. 41(5), 1031-7 (2002)

        v = self.vX + self.vM
        m = np.atleast_1d(m)
        mM, mX = self.vM * m, self.vX * m
        I = 0.5 * (mM * self.zM**2 + mX * self.zX**2)
        rootI = sqrt(I)
        try:
            if modified:
                b, β0, CMX, mmax = self.modified_coeffs
                Bϕ = β0
            else:
                b = 1.2
                try:
                    β0, β1, β2, CMX, mmax = self.regular_coeffs
                    α1, α2 = 1.4, 12
                    Bϕ = β0 + β1 * exp(-α1*rootI) + β2 * exp(-α2*rootI)
                except ValueError:
                    β0, β1, CMX, mmax = self.regular_coeffs
                    α = 2
                    Bϕ = β0 + β1 * exp(-α*rootI)

            z = abs(self.zM*self.zX)
            # Cϕ = 2 * CMX
            Cϕ = 2*np.sqrt(z) * CMX
            Φ = 1 - z*Aϕ*rootI/(1 + b*rootI)
            Φ += 2*(self.vM*self.vX/v) * Bϕ*m
            Φ += 2*(self.vM*self.vX)**1.5 / v * Cϕ*m**2
            Φ[m > mmax] = np.nan

            # Note definitions in Guggenheim: Π ⟨V1⟩ / RT = Φ ∑i ri where ri = ni/n1 = M1 × mi
            # are the solute-solvent mole ratios, with the 'standard' reference M1 being
            # the molar mass of solvent ('1' being the solvent).  Hence Π = Φ RT M1/⟨V1⟩ ∑i mi
            # = Φ RT ρ1 ∑i mi where ρ1 = M1/⟨V1⟩ ≈ ρw is the (average) solvent mass density.
            # The result here is expressed in bar = 10^5 Pa.

            Πid = NA*kB*T*ρw * v*m / atmospheric_pressure
            Π = Πid * Φ

        except:
            Φ = np.nan * np.empty_like(m)
            Πid = np.nan * np.empty_like(m)
            Π = np.nan * np.empty_like(m)

        result = np.vstack((m, Πid, Π, Φ)).transpose()

        x = m / (m + 1/water_molar_mass) # mole fraction

        return SimpleNamespace(m=m, Φ=Φ, Πid=Πid, Π=Π, x=x)

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(prog='pitzer',
                            description='evaluate osmotic coefficients via empirical models')
    parser.add_argument('-p', '--osmotic-pressure', action='store_true')
    parser.add_argument('-ex', '--excess', action='store_true')
    parser.add_argument('-m', '--molality', type=float, default=5.0)
    args = parser.parse_args()

    for name in ions:
        smiles = name_to_smiles[name]
        mol = Chem.MolFromSmiles(smiles)
        print(f'{name:>5}: {MolWt(mol):>8.4f} g/mol')

    T = 298 # temperature
    m = np.insert(np.geomspace(1e-3, args.molality, 1001), 0, 0.0) # prepend 0.0 to logspace

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(2*3.375, 2*3.375))

    for cation, ax in zip(cations, axes.ravel()):
        for anion in anions:

            model = PitzerModel(cation, anion)
            print(rf'{model.name:>6}: {model.zM} {model.zX} {model.vM} {model.vX}')

            for modified in [False, True]:
                result = model(T, m, modified)
                Π = result.Π
                Πex = result.Π - result.Πid
                if args.osmotic_pressure:
                    y = Πex if args.excess else Π
                else:
                    y = result.Φ - 1 if args.excess else result.Φ

                if modified:
                    ax.plot(result.m, y, ':', c=pl.get_color())
                else:
                    pl, = ax.plot(result.m, y, label=rf'{ce(anion)}')

        ax.text(0.975, 0.025, rf'{ce(cation)}',
                transform=ax.transAxes, ha='right', va='bottom')
        # ax.set_xlabel(r'mole fraction $x_z \equiv x_+ + x_-$')
        # ax.set_ylabel(r'osmotic\\coefficient $\phi$')
        ax.set_xlabel(r'molality $m$ / $\si{\mol\per\kilogram}$')
        if args.osmotic_pressure:
            if args.excess: ax.set_ylabel(r'$\Pi^\text{ex}$ / bar')
            else: ax.set_ylabel(r'$\Pi$ / bar')
        else:
            if args.excess: ax.set_ylabel(r'$\Phi^\text{ex}$')
            else: ax.set_ylabel(r'$\Phi$')
        ax.set_xlim([0, m[-1]])

    axes[0,0].legend(loc='best')

    plt.show()
