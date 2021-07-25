import numpy as np

# data taken from McGPU files (Badal A, Badano A. Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11):4878–80. )

#[COMPTON INTERACTIONS (relativistic impulse model with approximated one-electron analytical profiles)]
#[SHELL INFORMATION FROM COMMON/CGCO/: FCO, UICO, FJ0, KZCO, KSCO]

# Note: the FJ0 values represent the PENELOPE-2006 quantities: (J_{i,0} m_{e} c), and are thus dimensionless.

water_NUM_SHELLS = 3

water_compton_data = np.array([
	[  0.60000000E+01,  0.14000000E+02,  0.70756254E+02,   0,  30],
	[  0.20000000E+01,  0.24000000E+02,  0.79343843E+02,   8,   2],
	[  0.20000000E+01,  0.54300000E+03,  0.15485068E+02,   8,   1],
])
