import numpy as np

# data taken from McGPU files (Badal A, Badano A. Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11):4878–80. )

#[COMPTON INTERACTIONS (relativistic impulse model with approximated one-electron analytical profiles)]
#[SHELL INFORMATION FROM COMMON/CGCO/: FCO, UICO, FJ0, KZCO, KSCO]

# Note: the FJ0 values represent the PENELOPE-2006 quantities: (J_{i,0} m_{e} c), and are thus dimensionless.

bone_ICRP110_NUM_SHELLS = 17

bone_ICRP110_compton_data = np.array([
	[  0.20537205E+01,  0.00000000E+00,  0.16847848E+03,   0,  30],
	[  0.47308041E+01,  0.14109546E+02,  0.67158799E+02,   0,  30],
	[  0.24753264E+01,  0.24401994E+02,  0.66750759E+02,   0,  30],
	[  0.31871986E+00,  0.43918820E+02,  0.66847135E+02,   0,  30],
	[  0.11914448E-01,  0.73054922E+02,  0.50900249E+02,   0,  30],
	[  0.52551241E+00,  0.13620061E+03,  0.18313645E+02,   0,  30],
	[  0.91641772E+00,  0.26687851E+03,  0.23517076E+02,   0,  30],
	[  0.59518023E+00,  0.34700000E+03,  0.12360647E+02,  20,   4],
	[  0.29759012E+00,  0.35000000E+03,  0.12360647E+02,  20,   3],
	[  0.16789938E+00,  0.41000000E+03,  0.17814680E+02,   7,   1],
	[  0.29759012E+00,  0.43800000E+03,  0.23844264E+02,  20,   2],
	[  0.15679424E+01,  0.54300000E+03,  0.15485068E+02,   8,   1],
	[  0.73068001E-02,  0.10720000E+04,  0.11168434E+02,  11,   1],
	[  0.46076482E-02,  0.13030000E+04,  0.10209182E+02,  12,   1],
	[  0.16993213E+00,  0.21490000E+04,  0.81125311E+01,  15,   1],
	[  0.52386744E-02,  0.24720000E+04,  0.75780908E+01,  16,   1],
	[  0.29759012E+00,  0.40380000E+04,  0.60295840E+01,  20,   1],
])
