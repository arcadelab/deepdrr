import numpy as np

# data taken from McGPU files (Badal A, Badano A. Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11):4878–80. )

#[COMPTON INTERACTIONS (relativistic impulse model with approximated one-electron analytical profiles)]
#[SHELL INFORMATION FROM COMMON/CGCO/: FCO, UICO, FJ0, KZCO, KSCO]

# Note: the FJ0 values represent the PENELOPE-2006 quantities: (J_{i,0} m_{e} c), and are thus dimensionless.

connective_Woodard_NUM_SHELLS = 10

connective_Woodard_compton_data = np.array([
	[  0.74998747E+00,  0.00000000E+00,  0.22590042E+03,   0,  30],
	[  0.29092693E+01,  0.14031068E+02,  0.73319014E+02,   0,  30],
	[  0.85633140E+00,  0.24087048E+02,  0.78569662E+02,   0,  30],
	[  0.55966979E-02,  0.63000000E+02,  0.53444040E+02,  11,   2],
	[  0.39289088E+00,  0.27949316E+03,  0.20904241E+02,   0,  30],
	[  0.94921840E-01,  0.41000000E+03,  0.17814680E+02,   7,   1],
	[  0.83371408E+00,  0.54300000E+03,  0.15485068E+02,   8,   1],
	[  0.55966979E-02,  0.10720000E+04,  0.11168434E+02,  11,   1],
	[  0.40126017E-02,  0.24720000E+04,  0.75780908E+01,  16,   1],
	[  0.18146290E-02,  0.28230000E+04,  0.71258720E+01,  17,   1],
])
