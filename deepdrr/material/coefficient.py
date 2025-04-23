from dataclasses import dataclass


@dataclass
class CoefficientEntry:
    """
    A class to represent a coefficient entry for a material.
    Attributes:
        energy (float): Energy in MeV.
        mu_over_rho (float): Mass attenuation coefficient in cm^2/g.
        mu_en_over_rho (float): Mass energy-absorption
            coefficient in cm^2/g.
    """

    energy: float          # MeV
    mu_over_rho: float     # cm^2/g
    mu_en_over_rho: float  # cm^2/g
    