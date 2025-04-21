import os
from .material import Material

#### X-Ray Mass Attenuation Coefficients from NIST (https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients)
# Energy in MeV, Mass Attenuation Coef (\mu / \rho) [cm^2 / g], Mass Energy-Absorbition Coef (\mu_{en} / \rho) [cm^2 / g]

MATERIALS: dict[str, Material] = {}

_elemental_folder = os.path.join(os.path.dirname(__file__), "elemental_decompositions")
_material_folder = os.path.join(os.path.dirname(__file__), "material_decompositions")

# Load materials from the elemental folder
for file in os.listdir(_elemental_folder):
    if not file.endswith(".py") and not file.startswith("__"):
        symbol = file.split("_")[1]  # assumes format: Z_Symbol_Name
        path = os.path.join(_elemental_folder, file)
        MATERIALS[symbol] = Material(symbol, path=path)

# Load materials from the material folder
for file in os.listdir(_material_folder):
    if not file.endswith(".py") and not file.startswith("__"):
        name = file.split("_")[1]
        path = os.path.join(_material_folder, file)
        MATERIALS[name] = Material(name, path=path)

# Parity Dictionary Entries
MATERIALS["soft tissue"] = MATERIALS["tissue"]
MATERIALS["tissue_soft"] = MATERIALS["tissue"]
MATERIALS["iron"] = MATERIALS["Fe"]
MATERIALS["lead"] = MATERIALS["Pb"]
MATERIALS["copper"] = MATERIALS["Cu"]
MATERIALS["titanium"] = MATERIALS["Ti"]