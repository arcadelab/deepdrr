from . import ELEMENTS

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
