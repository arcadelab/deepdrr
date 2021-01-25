#
# TODO: transfer the material composition data from CONRAD project to here
#

element_to_Z = {
    "H": 1, "He":2, # TODO: CONTINUE
}

Z_to_element = {
    1: "H", 2: "He", # TODO: CONTINUE
}

AIR_COMPOSITION = {
    "H": 0.123456789, # TODO: get actual composition
    "C": 0.123456789,
    "N": 0.123456789,
    "O": 0.123456789,
    "Ar": 0.123456789
}

BONE_COMPOSITION = {
    "H": 0.123456789, # TODO: get actual composition
    "O": 0.123456789,
    "Ca": 0.123456789
}

material_compositions = {
    "bone": BONE_COMPOSITION,
    "soft tissue": None, # TODO: fill out this dictionary
    "air": AIR_COMPOSITION,
}
