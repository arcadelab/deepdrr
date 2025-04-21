import numpy as np
from .material import Material, CoefficientEntry
from .MATERIALS import MATERIALS

class Compound(Material):
    def __init__(self, name: str, materials: dict[Material, float]):
        """
        Initialize a Compound object with a name and materials.
        A compound is defined by its constituent materials and their respective fractions.
        The coefficients for the compound are calculated based on the materials provided.

        Args:
            name (str): Name of the compound.
            materials (dict[Material, float]): Dictionary of materials and their fractions.
        
        Example:
            compound = Compound("Water", {MATERIALS["H"]: 0.1119, MATERIALS["O"]: 0.8881})
        """
        super().__init__(name, [])
        self.materials = materials
        self.calculate_coefficients()

    def calculate_coefficients(self):
        """Calculate the coefficients for the compound based on its materials."""
        energy = np.array([])

        for material, _ in self.materials.items():
            if energy.size == 0:
                energy = MATERIALS[material].energy
            else:
                mask = ~np.isin(MATERIALS[material].energy, energy)
                energy = np.concatenate((energy, MATERIALS[material].energy[mask]))
        energy = np.sort(energy)
        
        mu_over_rho = np.zeros_like(energy)
        mu_en_over_rho = np.zeros_like(energy)

        for material, fraction in self.materials.items():
            material_coefficients = MATERIALS[material].lookup_list(energy)
            mu_over_rho = mu_over_rho + np.array([e.mu_over_rho for e in material_coefficients]) * fraction
            mu_en_over_rho = mu_en_over_rho + np.array([e.mu_en_over_rho for e in material_coefficients]) * fraction

        self.coefficients = [
            CoefficientEntry(e, mu_over_rho[i], mu_en_over_rho[i])
            for i, e in enumerate(energy)
        ]