import numpy as np

# data taken from McGPU files (Badal A, Badano A. Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11):4878–80. )

#[COMPTON INTERACTIONS (relativistic impulse model with approximated one-electron analytical profiles)]
#[SHELL INFORMATION FROM COMMON/CGCO/: FCO, UICO, FJ0, KZCO, KSCO]

# Note: the FJ0 values represent the PENELOPE-2006 quantities: (J_{i,0} m_{e} c), and are thus dimensionless.

air_NUM_SHELLS = 9

air_compton_data = np.array([
	[  0.47844272E+01,  0.14335344E+02,  0.66328116E+02,   0,  30],
	[  0.44018044E+00,  0.23936329E+02,  0.79148981E+02,   0,  30],
	[  0.18684440E-01,  0.24900000E+03,  0.14251744E+02,  18,   4],
	[  0.93422200E-02,  0.25100000E+03,  0.14251744E+02,  18,   3],
	[  0.30037400E-03,  0.28500000E+03,  0.20966508E+02,   6,  30],
	[  0.93422200E-02,  0.32600000E+03,  0.27133128E+02,  18,   2],
	[  0.15688600E+01,  0.41000000E+03,  0.17814680E+02,   7,   1],
	[  0.42149600E+00,  0.54300000E+03,  0.15485068E+02,   8,   1],
	[  0.93422200E-02,  0.32060000E+04,  0.67147640E+01,  18,   1],
])
