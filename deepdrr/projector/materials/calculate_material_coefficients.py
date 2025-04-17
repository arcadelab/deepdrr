import numpy as np
from .elements import ELEMENTS

def mass_attenuation_coefficient(energy: float, fractions: dict) -> float:
    """
    Calculate the mass attenuation coefficient (μ/ρ) of a compound/material.

    μ/ρ_mix = Σ [w_i * (μ/ρ)_i]  — weight-fraction average of element values

    Args:
        energy (float): Energy in MeV (e.g. 0.1)
        fractions (dict): Elemental mass fractions, e.g.:
                          {"H": 0.1119, "O": 0.8881} for water

    Returns:
        float: Compound mass attenuation coefficient in cm²/g

    Example:
        mu_rho = mass_attenuation_coefficient(0.1, {"H": 0.1119, "O": 0.8881})
    """
    return sum(
        fractions[symbol] * ELEMENTS[symbol].lookup(energy).mu_over_rho
        for symbol in fractions
    )


def mass_energy_absorption_coefficient(energy: float, fractions: dict) -> float:
    """
    Calculate the mass energy-absorption coefficient (μ_en/ρ) of a compound/material.

    μ_en/ρ_mix = Σ [w_i * (μ_en/ρ)_i] — weight-fraction average of element values

    Args:
        energy (float): Energy in MeV (e.g. 0.1)
        fractions (dict): Elemental mass fractions, e.g.:
                          {"H": 0.1119, "O": 0.8881} for water

    Returns:
        float: Compound mass energy-absorption coefficient in cm²/g

    Example:
        mu_en_rho = mass_energy_absorption_coefficient(0.1, {"H": 0.1119, "O": 0.8881})
    """
    return sum(
        fractions[symbol] * ELEMENTS[symbol].lookup(energy).mu_en_over_rho
        for symbol in fractions
    )


def calculate_material_coefficients(fractions: dict[str, float]) -> float:
    """
    # TODO which one to use? using lookup or using this?
    """
    elements = [ELEMENTS[elemental_symbol] for elemental_symbol in fractions]
    common_energies = elements[0].energy
    print(common_energies)
    for elem in elements:
        mask = ~np.isin(elem.energy, common_energies)
        common_energies = np.concatenate((common_energies, elem.energy[mask]))
    # print(f"before total energies:\n{common_energies}")
    common_energies = np.sort(common_energies)
    # print(f"after total energies:\n{common_energies}")

    mu_interp   = {}
    muen_interp = {}
    for symbol in fractions:
        energy = ELEMENTS[symbol].energy
        # print(f"before single energy list:\n{energy}")
        for j in range(1, len(energy)):
            if energy[j] == energy[j-1]:
                energy[j-1] *= (1 - 1e-9)  # tiny decrease for the first occurrence
        # print(f"after single energy list:\n{energy}")
        mu_interp[symbol] = np.exp(np.interp(np.log(common_energies), np.log(energy), np.log(ELEMENTS[symbol].mu_over_rho)))
        muen_interp[symbol] = np.exp(np.interp(np.log(common_energies), np.log(energy), np.log(ELEMENTS[symbol].mu_en_over_rho)))

    mu_compound   = np.zeros_like(common_energies)
    muen_compound = np.zeros_like(common_energies)
    for symbol in fractions:
        mu_compound   += fractions[symbol] * mu_interp[symbol]
        muen_compound += fractions[symbol] * muen_interp[symbol]

    return common_energies, mu_compound, muen_compound