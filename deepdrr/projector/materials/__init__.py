import os
from .elements import ElementData

ELEMENTS = {}

folder = os.path.join(os.path.dirname(__file__), "elemental_decompositions")

for file in os.listdir(folder):
    if not file.endswith(".py"):
        symbol = file.split("_")[1]  # assumes format: Z_Symbol_Name
        path = os.path.join(folder, file)
        with open(path, "r") as f:
            lines = f.readlines()
            element = ElementData(symbol)
            element.load_from_ascii(lines)
            ELEMENTS[symbol] = element

__all__ = ["ELEMENTS", "ElementData"]