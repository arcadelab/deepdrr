import numpy as np

# data taken from McGPU files (Badal A, Badano A. Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11):4878–80. )

#[COMPTON INTERACTIONS (relativistic impulse model with approximated one-electron analytical profiles)]
#[SHELL INFORMATION FROM COMMON/CGCO/: FCO, UICO, FJ0, KZCO, KSCO]

# Note: the FJ0 values represent the PENELOPE-2006 quantities: (J_{i,0} m_{e} c), and are thus dimensionless.

blood_ICRP110_NUM_SHELLS = 19

blood_ICRP110_compton_data = np.array([
	[  0.51350507E-02,  0.85823227E+01,  0.11888009E+03,   0,  30],
	[  0.33303013E+01,  0.13692650E+02,  0.73781786E+02,   0,  30],
	[  0.92385014E+00,  0.24031577E+02,  0.79207079E+02,   0,  30],
	[  0.22751155E-02,  0.62844459E+02,  0.42019376E+02,   0,  30],
	[  0.19516450E+00,  0.27828182E+03,  0.20897052E+02,   0,  30],
	[  0.20218676E-02,  0.29500000E+03,  0.13223974E+02,  19,   4],
	[  0.10109338E-02,  0.29700000E+03,  0.13223974E+02,  19,   3],
	[  0.10109338E-02,  0.37900000E+03,  0.25351660E+02,  19,   2],
	[  0.46560333E-01,  0.41000000E+03,  0.17814680E+02,   7,   1],
	[  0.92026034E+00,  0.54300000E+03,  0.15485068E+02,   8,   1],
	[  0.70774599E-03,  0.70800000E+03,  0.89073399E+01,  26,   4],
	[  0.35387299E-03,  0.72100000E+03,  0.89073399E+01,  26,   3],
	[  0.35387299E-03,  0.84800000E+03,  0.17403572E+02,  26,   2],
	[  0.85962354E-03,  0.10720000E+04,  0.11168434E+02,  11,   1],
	[  0.63804304E-03,  0.21490000E+04,  0.81125311E+01,  15,   1],
	[  0.12326293E-02,  0.24720000E+04,  0.75780908E+01,  16,   1],
	[  0.16723052E-02,  0.28230000E+04,  0.71258720E+01,  17,   1],
	[  0.10109338E-02,  0.36080000E+04,  0.63584704E+01,  19,   1],
	[  0.35387299E-03,  0.71110000E+04,  0.46044096E+01,  26,   1],
])
