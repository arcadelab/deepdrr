import os
import re
import csv
import numpy as np
from typing import List, Any
from numpy.typing import NDArray
from .log_interp import log_interp
from .coefficient import CoefficientEntry
from .mappings import element_map

#### X-Ray Mass Attenuation Coefficients from NIST (https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients)
# Energy in MeV, Mass Attenuation Coef (\mu / \rho) [cm^2 / g], Mass Energy-Absorbition Coef (\mu_{en} / \rho) [cm^2 / g]


class Material:
    """
    A class to represent a material with its coefficients.
    Attributes:
        name (str): Name of the material.
        coefficients (List[CoefficientEntry]): Coefficients for the material.
    """

    name: str
    coefficients: List[CoefficientEntry]

    _cache: dict[str, "Material"] = {}
    _material_dir = os.path.join(os.path.dirname(__file__), "material_decompositions")
    _custom_map: dict[str, str] = {}

    def __init__(self, name: str, coefficients: List[CoefficientEntry]):
        """
        DO NOT USE! Use `from_string` to initialize materials!

        Initializes a Material object with a name and coefficients.

        Args:
            name (str): Name of the material.
            coefficients (List[CoefficientEntry]): Coefficients for the material.
        Example:
            material = Material("Water", [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        """
        self.name: str = name
        self.coefficients = coefficients

    def __repr__(self):
        return f"Material(name={self.name}, coefficients={self.coefficients})"

    def __str__(self):
        return self.name

    def __hash__(self):
        """Return the hash of the material based on its name."""
        return hash(self.name)

    def __eq__(self, other):
        """Check equality based on the material name."""
        if not isinstance(other, Material):
            return NotImplemented
        return self.name == other.name

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

    def __array__(self) -> NDArray[Any]:
        """Return coefficients as a 2D numpy array."""
        return np.array(self.coefficients)

    def get(self, energy: float) -> CoefficientEntry:
        """Lookup coefficients for a given energy value.
        Args:
            energy (float): Energy value in MeV.
        Returns:
            CoefficientEntry: Coefficient entry for the given energy.
        """
        energies = self.energy
        for j in range(1, len(energies)):
            if energies[j] == energies[j - 1]:
                energies[j - 1] *= (
                    1 - 1e-9
                )  # tiny decrease for the first occurrence for proper interpolation
        mu_rho = float(log_interp(energy, energies, self.mu_over_rho))
        mu_en_rho = float(log_interp(energy, energies, self.mu_en_over_rho))
        return CoefficientEntry(energy, mu_rho, mu_en_rho)

    def get_list(self, energies: NDArray[Any]) -> List[CoefficientEntry]:
        """Lookup coefficients for a list of energies."""
        return [self.get(e) for e in energies]

    def get_coefficients(self, energy_keV: float) -> CoefficientEntry:
        """Returns the coefficients for the specified KeV energy level. Legacy function.
        Args:
            energy_keV: energy level of photon/ray (KeV)
        Returns:
            the interpolated coefficients (in [cm^2 / g])
        """
        # Convert MeV to keV
        energy = energy_keV / 1000
        return self.get(energy)

    @classmethod
    def register_map(cls, mapping: dict[str, str]):
        cls._custom_map.update(mapping)

    @classmethod
    def from_string(cls, name: str, compound_string: bool = False) -> "Material":
        """
        Create a Material instance from a string name or compound string.
        Args:
            name (str): Name of the material or compound string.
            compound_string (bool): If True, treat the name as a compound string.
        Returns:
            Material: Material instance.
        Example:
            material = Material.from_string("bone")
            material = Material.from_string("C") # Carbon
            material = Material.from_string("N0.758391O0.228770Cl0.012840", compound_string=True) # Air
        """
        # Check if custom mapping for name exists
        mapped_name = cls._custom_map.get(name, name)
        if mapped_name in cls._cache:
            return cls._cache[mapped_name]

        # 1. Handle custom string mapping (compound mode)
        if compound_string:
            # NOTE: this assumes decomposition like "H1119O8881" (for 11.19% H, 88.81% O)
            materials = cls._parse_compound_string(mapped_name)
            coefficients = cls.calc_material_coeffs_custom_compound(materials)
            instance = cls(mapped_name, coefficients=coefficients)
            cls._cache[mapped_name] = instance
            return instance

        # 2. Normal material name from file (default mode)
        path = os.path.join(cls._material_dir, mapped_name)
        if not os.path.isfile(path):
            raise AttributeError(f"Material '{mapped_name}' not found at {path}")

        lines = []
        with open(path, "r") as f:
            lines = f.readlines()

        instance = cls(mapped_name, coefficients=cls.load_material_coeffs_from_lines(lines))
        cls._cache[mapped_name] = instance
        return instance

    @staticmethod
    def _parse_compound_string(name: str) -> dict[str, float]:
        """
        Parse a compound string like 'H0.112O0.888' -> {'H': 0.112, 'O': 0.888}

        Only supports proper floats (e.g., H0.5, not H5 or H.5).
        """
        matches = re.findall(r"([A-Z][a-z]?)([0-9]+\.[0-9]+)", name)
        if not matches:
            raise ValueError(f"Invalid compound string: {name}")

        parsed = {elem: float(fraction) for elem, fraction in matches}
        total = sum(parsed.values())
        if not np.isclose(total, 1.0, atol=1e-4):
            raise ValueError(f"Fractions must sum to 1.0, got {total} for {name}")
        return parsed

    @staticmethod
    def load_material_coeffs_from_lines(lines: List[str]) -> List[CoefficientEntry]:
        """
        Load coefficients from list of lines. Each line should contain energy, mu_over_rho, and mu_en_over_rho values.

        Args:
            lines (List[str]): Lines of the file containing coefficients.
        Returns:
            List[CoefficientEntry]: List of CoefficientEntry objects.
        Example:
            coefficients = load_material_coeffs_from_lines(["Energy mu_rho mu_en_rho", "0.1 0.2 0.3", "0.4 0.5 0.6"])
        """
        coefficients: List[CoefficientEntry] = []
        for line in lines:
            if (
                line.strip()
                and not line.startswith("_")
                and not line.startswith("Energy")
            ):
                parts = line.strip().split()
                energy, mu_rho, mu_en_rho = map(float, parts)
                coefficients.append(CoefficientEntry(energy, mu_rho, mu_en_rho))
        return coefficients

    @staticmethod
    def calc_material_coeffs_custom_compound(
        materials: dict[str, float],
    ) -> List[CoefficientEntry]:
        """Calculate the coefficients for the compound based on its materials.

        Args:
            materials (dict[str, float]): Dictionary of materials and their fractions.

        Example:
            coefficients = calculate_custom_material_coefficients({Material.from_string("H"): 0.1119, Material.O: 0.8881})
        """
        energy = np.array([])

        for material, _ in materials.items():
            if energy.size == 0:
                energy = Material.from_string(material).energy
            else:
                mask = ~np.isin(Material.from_string(material).energy, energy)
                energy = np.concatenate(
                    (energy, Material.from_string(material).energy[mask])
                )
        energy = np.sort(energy)

        mu_over_rho = np.zeros_like(energy)
        mu_en_over_rho = np.zeros_like(energy)

        for material, fraction in materials.items():
            material_coefficients = Material.from_string(material).get_list(energy)
            mu_over_rho = (
                mu_over_rho
                + np.array([e.mu_over_rho for e in material_coefficients]) * fraction
            )
            mu_en_over_rho = (
                mu_en_over_rho
                + np.array([e.mu_en_over_rho for e in material_coefficients]) * fraction
            )

        return [
            CoefficientEntry(e, mu_over_rho[i], mu_en_over_rho[i])
            for i, e in enumerate(energy)
        ]

    @staticmethod
    def from_csv(path: str) -> None:
        """
        Load and register from materials.csv file in deepdrr.

        The materials.csv file should be structured like a typical DukeSim material decomposition.

        Args:
            path (str): Path to the materials.csv file.
        Example:
            from deepdrr.material import Material\n
            Material.from_csv("path/to/materials.csv")
        """

        # csv file should be structured like this:
        # Name,comment,MassDensity,H,He,C,N,O
        # Water,,1.0,0.1119,0.0,0.0,0.0,0.8881

        with open(path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row["Name"]
                composition = ""
                for key, val in row.items():
                    if key in ("Name", "MassDensity", "comment", "density_l", "density_u", "OrganID", "tissue_group", "Comment"):
                        continue
                    try:
                        val = float(val)
                    except ValueError: 
                        raise ValueError(
                            f"Invalid value for {key} in row {row}: {val}"
                        )
                    if val > 0.0:
                        composition += f"{key}{val:.6f}"
                # Register with DeepDRR
                Material.from_string(composition, compound_string=True)
                # Map the label name to the composition string
                Material.register_map({name: composition})


Material.register_map(element_map)
