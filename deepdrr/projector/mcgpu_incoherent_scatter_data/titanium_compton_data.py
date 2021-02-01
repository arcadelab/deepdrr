import numpy as np

# data taken from McGPU files (Badal A, Badano A. Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11):4878–80. )

#[COMPTON INTERACTIONS (relativistic impulse model with approximated one-electron analytical profiles)]
#[SHELL INFORMATION FROM COMMON/CGCO/: FCO, UICO, FJ0, KZCO, KSCO]

# Note: the FJ0 values represent the PENELOPE-2006 quantities: (J_{i,0} m_{e} c), and are thus dimensionless.

titanium_NUM_SHELLS = 7

titanium_compton_data = np.array([
	[  0.40000000E+01,  0.00000000E+00,  0.10266618E+03,   0,  30],
	[  0.60000000E+01,  0.33000000E+02,  0.38644152E+02,  22,  30],
	[  0.20000000E+01,  0.58000000E+02,  0.60432876E+02,  22,   5],
	[  0.40000000E+01,  0.45500000E+03,  0.10935473E+02,  22,   4],
	[  0.20000000E+01,  0.46100000E+03,  0.10935473E+02,  22,   3],
	[  0.20000000E+01,  0.56100000E+03,  0.21240580E+02,  22,   2],
	[  0.20000000E+01,  0.49660000E+04,  0.54677364E+01,  22,   1],
])
