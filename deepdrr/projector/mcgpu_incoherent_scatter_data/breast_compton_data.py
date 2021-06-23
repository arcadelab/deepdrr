import numpy as np

# data taken from McGPU files (Badal A, Badano A. Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11):4878–80. )

#[COMPTON INTERACTIONS (relativistic impulse model with approximated one-electron analytical profiles)]
#[SHELL INFORMATION FROM COMMON/CGCO/: FCO, UICO, FJ0, KZCO, KSCO]

# Note: the FJ0 values represent the PENELOPE-2006 quantities: (J_{i,0} m_{e} c), and are thus dimensionless.

breast_NUM_SHELLS = 8

breast_compton_data = np.array([
	[  0.15660458E+01,  0.00000000E+00,  0.15684097E+03,   0,  30],
	[  0.18910521E+01,  0.14014456E+02,  0.84930128E+02,   0,  30],
	[  0.41254345E+00,  0.23976949E+02,  0.79469405E+02,   0,  30],
	[  0.47547317E-02,  0.14875000E+03,  0.22371127E+02,  15,  30],
	[  0.78213140E+00,  0.28500000E+03,  0.20966508E+02,   6,  30],
	[  0.27337042E-01,  0.41000000E+03,  0.17814680E+02,   7,   1],
	[  0.41135476E+00,  0.54300000E+03,  0.15485068E+02,   8,   1],
	[  0.11886829E-02,  0.21490000E+04,  0.81125311E+01,  15,   1],
])
