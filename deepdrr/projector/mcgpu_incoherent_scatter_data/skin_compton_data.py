import numpy as np

# data taken from McGPU files (Badal A, Badano A. Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11):4878–80. )

#[COMPTON INTERACTIONS (relativistic impulse model with approximated one-electron analytical profiles)]
#[SHELL INFORMATION FROM COMMON/CGCO/: FCO, UICO, FJ0, KZCO, KSCO]

# Note: the FJ0 values represent the PENELOPE-2006 quantities: (J_{i,0} m_{e} c), and are thus dimensionless.

skin_ICRP110_NUM_SHELLS = 15

skin_ICRP110_compton_data = np.array([
	[  0.67258674E+00,  0.00000000E+00,  0.23443885E+03,   0,  30],
	[  0.27933137E+01,  0.14020112E+02,  0.73680473E+02,   0,  30],
	[  0.82990697E+00,  0.24009300E+02,  0.79100265E+02,   0,  30],
	[  0.17536320E-02,  0.63000000E+02,  0.53444040E+02,  11,   2],
	[  0.34843603E+00,  0.28116178E+03,  0.20926827E+02,   0,  30],
	[  0.10311525E-02,  0.29500000E+03,  0.13223974E+02,  19,   4],
	[  0.51557624E-03,  0.29700000E+03,  0.13223974E+02,  19,   3],
	[  0.51557624E-03,  0.37900000E+03,  0.25351660E+02,  19,   2],
	[  0.60443778E-01,  0.41000000E+03,  0.17814680E+02,   7,   1],
	[  0.81896994E+00,  0.54300000E+03,  0.15485068E+02,   8,   1],
	[  0.17536320E-02,  0.10720000E+04,  0.11168434E+02,  11,   1],
	[  0.65080390E-03,  0.21490000E+04,  0.81125311E+01,  15,   1],
	[  0.12572819E-02,  0.24720000E+04,  0.75780908E+01,  16,   1],
	[  0.17057513E-02,  0.28230000E+04,  0.71258720E+01,  17,   1],
	[  0.51557624E-03,  0.36080000E+04,  0.63584704E+01,  19,   1],
])
