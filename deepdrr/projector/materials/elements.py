import numpy as np
from dataclasses import dataclass
from typing import List

# Coefficients are taken from https://physics.nist.gov/PhysRefData/XrayMassCoef/tab3.html

@dataclass
class CoefficientEntry:
    energy: float          # MeV
    mu_over_rho: float     # cm^2/g
    mu_en_over_rho: float  # cm^2/g

class ElementData:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.entries: List[CoefficientEntry] = []

    def load_from_ascii(self, lines: List[str]):
        for line in lines:
            if line.strip() and not line.startswith('_') and not line.startswith('Energy'):
                parts = line.strip().split()
                energy, mu_rho, mu_en_rho = map(float, parts)
                self.entries.append(CoefficientEntry(energy, mu_rho, mu_en_rho))

    def lookup(self, energy: float) -> CoefficientEntry:
        energies = [e.energy for e in self.entries]
        idx = np.searchsorted(energies, energy)
        if idx == 0:
            return self.entries[0]
        elif idx >= len(self.entries):
            return self.entries[-1]

        e1, e2 = self.entries[idx - 1], self.entries[idx]
        # Linear interpolation
        factor = (energy - e1.energy) / (e2.energy - e1.energy)
        mu_rho = e1.mu_over_rho + factor * (e2.mu_over_rho - e1.mu_over_rho)
        mu_en_rho = e1.mu_en_over_rho + factor * (e2.mu_en_over_rho - e1.mu_en_over_rho)
        return CoefficientEntry(energy, mu_rho, mu_en_rho)

    def as_array(self) -> np.ndarray:
        return np.array([
            [e.energy, e.mu_over_rho, e.mu_en_over_rho]
            for e in self.entries
        ])