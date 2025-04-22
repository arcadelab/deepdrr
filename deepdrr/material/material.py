import os
import re
import numpy as np
from dataclasses import dataclass
from typing import List, Any
from numpy.typing import NDArray
from .log_interp import log_interp

#### X-Ray Mass Attenuation Coefficients from NIST (https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients)
# Energy in MeV, Mass Attenuation Coef (\mu / \rho) [cm^2 / g], Mass Energy-Absorbition Coef (\mu_{en} / \rho) [cm^2 / g]

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

class MaterialMeta(type):
    _cache: dict[str, "Material"] = {}
    _material_dir = os.path.join(os.path.dirname(__file__), "material_decompositions")
    _custom_map: dict[str, str] = {}

    def __getattr__(cls, name: str):
        return cls.from_string(name)
    
    def register_map(cls, mapping: dict[str, str]):
        cls._custom_map.update(mapping)
    
    def from_string(cls, name: str) -> "Material":
        if name in cls._cache:
            return cls._cache[name]
        
        # 1. Handle custom string mapping (compound mode)
        if cls._is_compound_string(name):
            # NOTE: this assumes decomposition like "H1119O8881" (for 11.19% H, 88.81% O)
            materials = cls._parse_compound_string(name)
            coefficients = cls.calc_material_coeffs_custom_compound(materials)
            instance = cls(name, coefficients=coefficients)
            cls._cache[name] = instance
            return instance

        # 2. Normal material name from file (default mode)
        mapped_name = cls._custom_map.get(name, name) # Check if custom mapping exists
        path = os.path.join(cls._material_dir, mapped_name)
        if not os.path.isfile(path):
            raise AttributeError(f"Material '{name}' not found at {path}")
        
        lines = []
        with open(path, "r") as f:
            lines = f.readlines()

        instance = cls(name, coefficients=cls.load_material_coeffs_from_lines(lines))
        cls._cache[name] = instance
        return instance

    @staticmethod
    def _is_compound_string(name: str) -> bool:
        return any(c.isdigit() for c in name)
    
    @staticmethod
    def _parse_compound_string(name: str) -> dict[str, float]:
        """
        Parse a compound string like "H1119O8881" -> {"H": 0.1119, "O": 0.8881}
        """
        # Matches chemical symbols with their percentage ie. H1119, O8881, C12,...
        matches = re.findall(r'([A-Z][a-z]?)(\d+)', name)
        if not matches:
            raise ValueError(f"Invalid compound string: {name}")

        parsed = {elem: int(percent) / 10000 for elem, percent in matches}
        total = sum(parsed.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"Fractions must sum to 1.0, got {total}")
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
            _ = load_material_coeffs_from_lines(["Energy mu_rho mu_en_rho", "0.1 0.2 0.3", "0.4 0.5 0.6"])
        """
        coefficients: List[CoefficientEntry] = []
        for line in lines:
            if line.strip() and not line.startswith('_') and not line.startswith('Energy'):
                parts = line.strip().split()
                energy, mu_rho, mu_en_rho = map(float, parts)
                coefficients.append(CoefficientEntry(energy, mu_rho, mu_en_rho))
        return coefficients
    
    @staticmethod
    def calc_material_coeffs_custom_compound(materials: dict[str, float]) -> List[CoefficientEntry]:
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
                energy = np.concatenate((energy, Material.from_string(material).energy[mask]))
        energy = np.sort(energy)
        
        mu_over_rho = np.zeros_like(energy)
        mu_en_over_rho = np.zeros_like(energy)

        for material, fraction in materials.items():
            material_coefficients = Material.from_string(material).get_list(energy)
            mu_over_rho = mu_over_rho + np.array([e.mu_over_rho for e in material_coefficients]) * fraction
            mu_en_over_rho = mu_en_over_rho + np.array([e.mu_en_over_rho for e in material_coefficients]) * fraction

        return [
            CoefficientEntry(e, mu_over_rho[i], mu_en_over_rho[i])
            for i, e in enumerate(energy)
        ]

class Material(metaclass=MaterialMeta):
    """
    A class to represent a material with its coefficients.
    Attributes:
        name (str): Name of the material.
        coefficients (List[CoefficientEntry]): Coefficients for the material.
    """
    name: str
    coefficients: List[CoefficientEntry]

    def __init__(self, name: str, coefficients: List[CoefficientEntry]):
        """
        Initialize a Material object with a name and coefficients.

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

    def as_array(self) -> NDArray[Any]:
        """Return coefficients as a 2D numpy array."""
        return np.array([
            [e.energy, e.mu_over_rho, e.mu_en_over_rho]
            for e in self.coefficients
        ])

    def get(self, energy: float) -> CoefficientEntry:
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
    
Material.register_map({
    "H":  "1_H_Hydrogen",
    "He": "2_He_Helium",
    "Li": "3_Li_Lithium",
    "Be": "4_Be_Beryllium",
    "B":  "5_B_Boron",
    "C":  "6_C_Carbon, Graphite",
    "N":  "7_N_Nitrogen",
    "O":  "8_O_Oxygen",
    "F":  "9_F_Fluorine",
    "Ne": "10_Ne_Neon",
    "Na": "11_Na_Sodium",
    "Mg": "12_Mg_Magnesium",
    "Al": "13_Al_Aluminum",
    "Si": "14_Si_Silicon",
    "P":  "15_P_Phosphorus",
    "S":  "16_S_Sulfur",
    "Cl": "17_Cl_Chlorine",
    "Ar": "18_Ar_Argon",
    "K":  "19_K_Potassium",
    "Ca": "20_Ca_Calcium",
    "Sc": "21_Sc_Scandium",
    "Ti": "22_Ti_Titanium",
    "V":  "23_V_Vanadium",
    "Cr": "24_Cr_Chromium",
    "Mn": "25_Mn_Manganese",
    "Fe": "26_Fe_Iron",
    "Co": "27_Co_Cobalt",
    "Ni": "28_Ni_Nickel",
    "Cu": "29_Cu_Copper",
    "Zn": "30_Zn_Zinc",
    "Ga": "31_Ga_Gallium",
    "Ge": "32_Ge_Germanium",
    "As": "33_As_Arsenic",
    "Se": "34_Se_Selenium",
    "Br": "35_Br_Bromine",
    "Kr": "36_Kr_Krypton",
    "Rb": "37_Rb_Rubidium",
    "Sr": "38_Sr_Strontium",
    "Y":  "39_Y_Yttrium",
    "Zr": "40_Zr_Zirconium",
    "Nb": "41_Nb_Niobium",
    "Mo": "42_Mo_Molybdenum",
    "Tc": "43_Tc_Technetium",
    "Ru": "44_Ru_Ruthenium",
    "Rh": "45_Rh_Rhodium",
    "Pd": "46_Pd_Palladium",
    "Ag": "47_Ag_Silver",
    "Cd": "48_Cd_Cadmium",
    "In": "49_In_Indium",
    "Sn": "50_Sn_Tin",
    "Sb": "51_Sb_Antimony",
    "Te": "52_Te_Tellurium",
    "I":  "53_I_Iodine",
    "Xe": "54_Xe_Xenon",
    "Cs": "55_Cs_Cesium",
    "Ba": "56_Ba_Barium",
    "La": "57_La_Lanthanum",
    "Ce": "58_Ce_Cerium",
    "Pr": "59_Pr_Praseodymium",
    "Nd": "60_Nd_Neodymium",
    "Pm": "61_Pm_Promethium",
    "Sm": "62_Sm_Samarium",
    "Eu": "63_Eu_Europium",
    "Gd": "64_Gd_Gadolinium",
    "Tb": "65_Tb_Terbium",
    "Dy": "66_Dy_Dysprosium",
    "Ho": "67_Ho_Holmium",
    "Er": "68_Er_Erbium",
    "Tm": "69_Tm_Thulium",
    "Yb": "70_Yb_Ytterbium",
    "Lu": "71_Lu_Lutetium",
    "Hf": "72_Hf_Hafnium",
    "Ta": "73_Ta_Tantalum",
    "W":  "74_W_Tungsten",
    "Re": "75_Re_Rhenium",
    "Os": "76_Os_Osmium",
    "Ir": "77_Ir_Iridium",
    "Pt": "78_Pt_Platinum",
    "Au": "79_Au_Gold",
    "Hg": "80_Hg_Mercury",
    "Tl": "81_Tl_Thallium",
    "Pb": "82_Pb_Lead",
    "Bi": "83_Bi_Bismuth",
    "Po": "84_Po_Polonium",
    "At": "85_At_Astatine",
    "Rn": "86_Rn_Radon",
    "Fr": "87_Fr_Francium",
    "Ra": "88_Ra_Radium",
    "Ac": "89_Ac_Actinium",
    "Th": "90_Th_Thorium",
    "Pa": "91_Pa_Protactinium",
    "U":  "92_U_Uranium",

    # Add named materials too
    "soft tissue": "tissue",
    "tissue_soft": "tissue",
    "iron":        "26_Fe_Iron",
    "lead":        "82_Pb_Lead",
    "copper":      "29_Cu_Copper",
    "titanium":    "22_Ti_Titanium",
})
