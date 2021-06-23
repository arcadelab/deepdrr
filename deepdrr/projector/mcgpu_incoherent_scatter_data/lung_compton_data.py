import numpy as np

# data taken from McGPU files (Badal A, Badano A. Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11):4878–80. )

#[COMPTON INTERACTIONS (relativistic impulse model with approximated one-electron analytical profiles)]
#[SHELL INFORMATION FROM COMMON/CGCO/: FCO, UICO, FJ0, KZCO, KSCO]

# Note: the FJ0 values represent the PENELOPE-2006 quantities: (J_{i,0} m_{e} c), and are thus dimensionless.

lung_ICRP110_NUM_SHELLS = 17

lung_ICRP110_compton_data = np.array([
	[  0.13518371E-02,  0.60000000E+01,  0.30345383E+03,   0,  30],
	[  0.32983624E+01,  0.13690285E+02,  0.73878827E+02,   0,  30],
	[  0.92331724E+00,  0.24020372E+02,  0.79016600E+02,   0,  30],
	[  0.17025554E-02,  0.63000000E+02,  0.53444040E+02,  11,   2],
	[  0.17346967E-01,  0.17702404E+03,  0.19535232E+02,   0,  30],
	[  0.16560692E-02,  0.27000000E+03,  0.29188668E+02,  17,   2],
	[  0.17434714E+00,  0.28500000E+03,  0.20966508E+02,   6,  30],
	[  0.20022378E-02,  0.29500000E+03,  0.13223974E+02,  19,   4],
	[  0.10011189E-02,  0.29700000E+03,  0.13223974E+02,  19,   3],
	[  0.10011189E-02,  0.37900000E+03,  0.25351660E+02,  19,   2],
	[  0.44711070E-01,  0.41000000E+03,  0.17814680E+02,   7,   1],
	[  0.91254903E+00,  0.54300000E+03,  0.15485068E+02,   8,   1],
	[  0.17025554E-02,  0.10720000E+04,  0.11168434E+02,  11,   1],
	[  0.12636969E-02,  0.21490000E+04,  0.81125311E+01,  15,   1],
	[  0.18309930E-02,  0.24720000E+04,  0.75780908E+01,  16,   1],
	[  0.16560692E-02,  0.28230000E+04,  0.71258720E+01,  17,   1],
	[  0.10011189E-02,  0.36080000E+04,  0.63584704E+01,  19,   1],
])
