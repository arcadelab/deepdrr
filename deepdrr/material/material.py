import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Any
from numpy.typing import NDArray
from .log_interp import log_interp

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

class Material:
    def __init__(self, name: str, coefficients: Optional[List[CoefficientEntry] | NDArray[Any]] = None, path: Optional[str] = None):
        """
        Initialize a Material object with a name and coefficients.
        Coefficients can be provided as a list of CoefficientEntry objects, a 2D numpy array, or loaded from a file.
        Args:
            name (str): Name of the material.
            coefficients (List[CoefficientEntry] | NDArray[Any], optional): Coefficients for the material. Defaults to None.
            path (str, optional): Path to a file containing coefficients. Defaults to None. If coefficients are provided, this argument is ignored.
        Example:
            material = Material("Water", [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            material = Material("Water", np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
            material = Material("Water", path="path/to/file")
        """
        self.name: str = name
        self.coefficients: List[CoefficientEntry] = []

        if coefficients is not None:
            if isinstance(coefficients, list):
                self.coefficients = coefficients
            else:
                if coefficients.ndim == 1:
                    coefficients = coefficients.reshape(-1, 3)
                self.coefficients = [CoefficientEntry(*row) for row in coefficients]
        elif path is not None:
            with open(path, "r") as f:
                lines = f.readlines()
                self.load_from_ascii(lines)

    def __repr__(self):
        return f"Material(name={self.name}, coefficients={self.coefficients})"

    def __str__(self):
        return self.name
    
    @property
    def energy(self) -> NDArray[Any]:
        """Return the energy values of the coefficients."""
        return np.array([e.energy for e in self.coefficients])

    @property
    def mu_over_rho(self) -> NDArray[Any]:
        """Return the mass attenuation coefficient."""
        return np.array([e.mu_over_rho for e in self.coefficients])

    @property
    def mu_en_over_rho(self) -> NDArray[Any]:
        """Return the mass energy-absorption coefficient."""
        return np.array([e.mu_en_over_rho for e in self.coefficients])
    
    def load_from_ascii(self, lines: List[str]):
        """
        Load coefficients from an ASCII file. Clears current coefficients list.
        The file should contain lines with energy, mu_over_rho, and mu_en_over_rho values.

        Args:
            lines (List[str]): Lines from the file.
        Example:
            material.load_from_ascii(["Energy  mu_rho  mu_en_rho", "0.1  0.2  0.3", "0.4  0.5  0.6"])
        """
        self.coefficients = []
        for line in lines:
            if line.strip() and not line.startswith('_') and not line.startswith('Energy'):
                parts = line.strip().split()
                energy, mu_rho, mu_en_rho = map(float, parts)
                self.coefficients.append(CoefficientEntry(energy, mu_rho, mu_en_rho))

    def as_array(self) -> NDArray[Any]:
        """Return coefficients as a 2D numpy array."""
        return np.array([
            [e.energy, e.mu_over_rho, e.mu_en_over_rho]
            for e in self.coefficients
        ])

    def lookup(self, energy: float) -> CoefficientEntry:
        """Lookup coefficients for a given energy value.
        Args:
            energy (float): Energy value in MeV.
        Returns:
            CoefficientEntry: Coefficient entry for the given energy.
        """
        energies = self.energy
        for j in range(1, len(energies)):
            if energies[j] == energies[j-1]:
                energies[j-1] *= (1 - 1e-9)  # tiny decrease for the first occurrence for proper interpolation    
        mu_rho = float(log_interp(energy, energies, self.mu_over_rho))
        mu_en_rho = float(log_interp(energy, energies, self.mu_en_over_rho))
        return CoefficientEntry(energy, mu_rho, mu_en_rho)
    
    def lookup_list(self, energies: NDArray[Any]) -> List[CoefficientEntry]:
        """Lookup coefficients for a list of energies."""
        return [self.lookup(e) for e in energies]
    
    def get_coefficients(self, energy_keV: float) -> CoefficientEntry:
        """Returns the coefficients for the specified KeV energy level. Legacy function.
        Args:
            energy_keV: energy level of photon/ray (KeV)
        Returns:
            the interpolated coefficients (in [cm^2 / g])
        """
        # Convert MeV to keV
        energy = energy_keV / 1000
        return self.lookup(energy)