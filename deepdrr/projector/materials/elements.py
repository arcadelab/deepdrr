import numpy as np
from dataclasses import dataclass
from typing import List
import os

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

    @property
    def energy(self) -> np.ndarray:
        return np.array([e.energy for e in self.entries])

    @property
    def mu_over_rho(self) -> np.ndarray:
        return np.array([e.mu_over_rho for e in self.entries])

    @property
    def mu_en_over_rho(self) -> np.ndarray:
        return np.array([e.mu_en_over_rho for e in self.entries])


    def load_from_ascii(self, lines: List[str]):
        for line in lines:
            if line.strip() and not line.startswith('_') and not line.startswith('Energy'):
                parts = line.strip().split()
                energy, mu_rho, mu_en_rho = map(float, parts)
                self.entries.append(CoefficientEntry(energy, mu_rho, mu_en_rho))

    def as_array(self) -> np.ndarray:
        return np.array([
            [e.energy, e.mu_over_rho, e.mu_en_over_rho]
            for e in self.entries
        ])

    def lookup(self, energy: float) -> CoefficientEntry:
        energies = [e.energy for e in self.entries]
        idx = np.searchsorted(energies, energy)
        if idx == 0:
            return self.entries[0]
        elif idx >= len(self.entries):
            return self.entries[-1]

        e1, e2 = self.entries[idx - 1], self.entries[idx]
        # TODO check if this is correct at all? This was just very assumptious to do
        # Linear interpolation
        factor = (energy - e1.energy) / (e2.energy - e1.energy)
        mu_rho = e1.mu_over_rho + factor * (e2.mu_over_rho - e1.mu_over_rho)
        mu_en_rho = e1.mu_en_over_rho + factor * (e2.mu_en_over_rho - e1.mu_en_over_rho)
        return CoefficientEntry(energy, mu_rho, mu_en_rho)
    
ELEMENTS: dict[str, ElementData] = {}

folder = os.path.join(os.path.dirname(__file__), "elemental_decompositions")

for file in os.listdir(folder):
    if not file.endswith(".py") and not file.startswith("__"):
        symbol = file.split("_")[1]  # assumes format: Z_Symbol_Name
        path = os.path.join(folder, file)
        with open(path, "r") as f:
            lines = f.readlines()
            element = ElementData(symbol)
            element.load_from_ascii(lines)
            ELEMENTS[symbol] = element