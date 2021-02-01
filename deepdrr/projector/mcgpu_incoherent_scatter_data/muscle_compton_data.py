import numpy as np

# data taken from McGPU files (Badal A, Badano A. Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11):4878–80. )

#[COMPTON INTERACTIONS (relativistic impulse model with approximated one-electron analytical profiles)]
#[SHELL INFORMATION FROM COMMON/CGCO/: FCO, UICO, FJ0, KZCO, KSCO]

# Note: the FJ0 values represent the PENELOPE-2006 quantities: (J_{i,0} m_{e} c), and are thus dimensionless.

muscle_ICRP110_NUM_SHELLS = 17

muscle_ICRP110_compton_data = np.array([
	[  0.14407456E-02,  0.60000000E+01,  0.32116482E+03,   0,  30],
	[  0.33538725E+01,  0.13591073E+02,  0.74923897E+02,   0,  30],
	[  0.88948565E+00,  0.24000623E+02,  0.79022202E+02,   0,  30],
	[  0.85962354E-03,  0.63000000E+02,  0.53444040E+02,  11,   2],
	[  0.14172425E-01,  0.17144452E+03,  0.20523387E+02,   0,  30],
	[  0.55743506E-03,  0.27000000E+03,  0.29188668E+02,  17,   2],
	[  0.23364498E+00,  0.28500000E+03,  0.20966508E+02,   6,  30],
	[  0.40437353E-02,  0.29500000E+03,  0.13223974E+02,  19,   4],
	[  0.20218676E-02,  0.29700000E+03,  0.13223974E+02,  19,   3],
	[  0.20218676E-02,  0.37900000E+03,  0.25351660E+02,  19,   2],
	[  0.47971252E-01,  0.41000000E+03,  0.17814680E+02,   7,   1],
	[  0.87826188E+00,  0.54300000E+03,  0.15485068E+02,   8,   1],
	[  0.85962354E-03,  0.10720000E+04,  0.11168434E+02,  11,   1],
	[  0.12760861E-02,  0.21490000E+04,  0.81125311E+01,  15,   1],
	[  0.18489439E-02,  0.24720000E+04,  0.75780908E+01,  16,   1],
	[  0.55743506E-03,  0.28230000E+04,  0.71258720E+01,  17,   1],
	[  0.20218676E-02,  0.36080000E+04,  0.63584704E+01,  19,   1],
])
