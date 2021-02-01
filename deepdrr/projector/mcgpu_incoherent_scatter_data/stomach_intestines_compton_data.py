import numpy as np

# data taken from McGPU files (Badal A, Badano A. Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11):4878–80. )

#[COMPTON INTERACTIONS (relativistic impulse model with approximated one-electron analytical profiles)]
#[SHELL INFORMATION FROM COMMON/CGCO/: FCO, UICO, FJ0, KZCO, KSCO]

# Note: the FJ0 values represent the PENELOPE-2006 quantities: (J_{i,0} m_{e} c), and are thus dimensionless.

stomach_intestines_ICRP110_NUM_SHELLS = 17

stomach_intestines_ICRP110_compton_data = np.array([
	[  0.66304393E-03,  0.60000000E+01,  0.30345383E+03,   0,  30],
	[  0.32560798E+01,  0.13672059E+02,  0.74153683E+02,   0,  30],
	[  0.90551927E+00,  0.24008394E+02,  0.79185674E+02,   0,  30],
	[  0.83506286E-03,  0.63000000E+02,  0.53444040E+02,  11,   2],
	[  0.81231256E-02,  0.17887578E+03,  0.18986357E+02,   0,  30],
	[  0.10830167E-02,  0.27000000E+03,  0.29188668E+02,  17,   2],
	[  0.18221487E+00,  0.28500000E+03,  0.20966508E+02,   6,  30],
	[  0.98204999E-03,  0.29500000E+03,  0.13223974E+02,  19,   4],
	[  0.49102499E-03,  0.29700000E+03,  0.13223974E+02,  19,   3],
	[  0.49102499E-03,  0.37900000E+03,  0.25351660E+02,  19,   2],
	[  0.34265180E-01,  0.41000000E+03,  0.17814680E+02,   7,   1],
	[  0.89996696E+00,  0.54300000E+03,  0.15485068E+02,   8,   1],
	[  0.83506286E-03,  0.10720000E+04,  0.11168434E+02,  11,   1],
	[  0.61981324E-03,  0.21490000E+04,  0.81125311E+01,  15,   1],
	[  0.59870565E-03,  0.24720000E+04,  0.75780908E+01,  16,   1],
	[  0.10830167E-02,  0.28230000E+04,  0.71258720E+01,  17,   1],
	[  0.49102499E-03,  0.36080000E+04,  0.63584704E+01,  19,   1],
])
