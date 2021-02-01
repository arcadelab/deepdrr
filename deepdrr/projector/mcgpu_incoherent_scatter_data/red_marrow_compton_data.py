import numpy as np

# data taken from McGPU files (Badal A, Badano A. Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11):4878–80. )

#[COMPTON INTERACTIONS (relativistic impulse model with approximated one-electron analytical profiles)]
#[SHELL INFORMATION FROM COMMON/CGCO/: FCO, UICO, FJ0, KZCO, KSCO]

# Note: the FJ0 values represent the PENELOPE-2006 quantities: (J_{i,0} m_{e} c), and are thus dimensionless.

red_marrow_Woodard_NUM_SHELLS = 20

red_marrow_Woodard_compton_data = np.array([
	[  0.13286460E+01,  0.00000000E+00,  0.16745537E+03,   0,  30],
	[  0.21727705E+01,  0.14020201E+02,  0.80655970E+02,   0,  30],
	[  0.53360910E+00,  0.23947695E+02,  0.79341985E+02,   0,  30],
	[  0.10312870E-02,  0.53000000E+02,  0.30559028E+02,  26,  30],
	[  0.10861710E-01,  0.17648436E+03,  0.20240845E+02,   0,  30],
	[  0.10830167E-02,  0.27000000E+03,  0.29188668E+02,  17,   2],
	[  0.66172770E+00,  0.28500000E+03,  0.20966508E+02,   6,  30],
	[  0.19641000E-02,  0.29500000E+03,  0.13223974E+02,  19,   4],
	[  0.98204999E-03,  0.29700000E+03,  0.13223974E+02,  19,   3],
	[  0.98204999E-03,  0.37900000E+03,  0.25351660E+02,  19,   2],
	[  0.46600645E-01,  0.41000000E+03,  0.17814680E+02,   7,   1],
	[  0.52678066E+00,  0.54300000E+03,  0.15485068E+02,   8,   1],
	[  0.68752467E-03,  0.70800000E+03,  0.89073399E+01,  26,   4],
	[  0.34376234E-03,  0.72100000E+03,  0.89073399E+01,  26,   3],
	[  0.34376234E-03,  0.84800000E+03,  0.17403572E+02,  26,   2],
	[  0.61981324E-03,  0.21490000E+04,  0.81125311E+01,  15,   1],
	[  0.11974113E-02,  0.24720000E+04,  0.75780908E+01,  16,   1],
	[  0.10830167E-02,  0.28230000E+04,  0.71258720E+01,  17,   1],
	[  0.98204999E-03,  0.36080000E+04,  0.63584704E+01,  19,   1],
	[  0.34376234E-03,  0.71110000E+04,  0.46044096E+01,  26,   1],
])
