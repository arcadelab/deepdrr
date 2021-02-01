import numpy as np

# data taken from McGPU files (Badal A, Badano A. Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11):4878–80. )

#[COMPTON INTERACTIONS (relativistic impulse model with approximated one-electron analytical profiles)]
#[SHELL INFORMATION FROM COMMON/CGCO/: FCO, UICO, FJ0, KZCO, KSCO]

# Note: the FJ0 values represent the PENELOPE-2006 quantities: (J_{i,0} m_{e} c), and are thus dimensionless.

adipose_ICRP110_NUM_SHELLS = 10

adipose_ICRP110_compton_data = np.array([
	[  0.17327811E+01,  0.00000000E+00,  0.14653147E+03,   0,  30],
	[  0.16608927E+01,  0.14005330E+02,  0.89487767E+02,   0,  30],
	[  0.32055649E+00,  0.24027289E+02,  0.79073911E+02,   0,  30],
	[  0.76913685E-03,  0.63000000E+02,  0.53444040E+02,  11,   2],
	[  0.86984763E+00,  0.28458134E+03,  0.20960168E+02,   0,  30],
	[  0.10099211E-01,  0.41000000E+03,  0.17814680E+02,   7,   1],
	[  0.31719888E+00,  0.54300000E+03,  0.15485068E+02,   8,   1],
	[  0.76913685E-03,  0.10720000E+04,  0.11168434E+02,  11,   1],
	[  0.55143941E-03,  0.24720000E+04,  0.75780908E+01,  16,   1],
	[  0.49875768E-03,  0.28230000E+04,  0.71258720E+01,  17,   1],
])
