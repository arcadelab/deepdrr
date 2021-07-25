import numpy as np

# data taken from McGPU files (Badal A, Badano A. Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11):4878–80. )

#[COMPTON INTERACTIONS (relativistic impulse model with approximated one-electron analytical profiles)]
#[SHELL INFORMATION FROM COMMON/CGCO/: FCO, UICO, FJ0, KZCO, KSCO]

# Note: the FJ0 values represent the PENELOPE-2006 quantities: (J_{i,0} m_{e} c), and are thus dimensionless.

cartilage_ICRP110_NUM_SHELLS = 13

cartilage_ICRP110_compton_data = np.array([
	[  0.22833750E-02,  0.60000000E+01,  0.28366452E+03,  11,   5],
	[  0.34409355E+01,  0.13678888E+02,  0.73249685E+02,   0,  30],
	[  0.99194123E+00,  0.24085933E+02,  0.78706727E+02,   0,  30],
	[  0.45667500E-02,  0.63000000E+02,  0.53444040E+02,  11,   2],
	[  0.88561532E-01,  0.16032631E+03,  0.21428608E+02,   0,  30],
	[  0.17768243E-02,  0.27000000E+03,  0.29188668E+02,  17,   2],
	[  0.17307416E+00,  0.28500000E+03,  0.20966508E+02,   6,  30],
	[  0.32980236E-01,  0.41000000E+03,  0.17814680E+02,   7,   1],
	[  0.97646415E+00,  0.54300000E+03,  0.15485068E+02,   8,   1],
	[  0.45667500E-02,  0.10720000E+04,  0.11168434E+02,  11,   1],
	[  0.14914256E-01,  0.21490000E+04,  0.81125311E+01,  15,   1],
	[  0.58935087E-02,  0.24720000E+04,  0.75780908E+01,  16,   1],
	[  0.17768243E-02,  0.28230000E+04,  0.71258720E+01,  17,   1],
])
