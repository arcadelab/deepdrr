# `deepdrr`

Forked and detached from [DeepDRR](https://github.com/mathiasunberath/DeepDRR), `deepdrr` has been upgraded with new features and an improved focus on usability.

`deepdrr` is in active development.

## Getting Started
### Installation

`deepdrr` is `pip`-installable, but it is not yet listed on the Python Package Index. Currently, we recommend installing from source inside a conda or virtualenv Python environment.

```bash
git clone https://github.com/arcadelab/deepdrr.git
pip install -e deepdrr
```

### Usage

For example usage, run
```bash
python example_projector.py
```

More detailed use-cases are pending.

## Documentation

Documentation efforts are pending. Much of the code has been 


## Reference

We hope this proves useful for medical imaging research. If you use our work, we would kindly ask you to reference our work. 
The MICCAI article covers the basic DeepDRR pipeline and task-based evaluation:
```
@inproceedings{DeepDRR2018,
  author       = {Unberath, Mathias and Zaech, Jan-Nico and Lee, Sing Chun and Bier, Bastian and Fotouhi, Javad and Armand, Mehran and Navab, Nassir},
  title        = {{DeepDRR--A Catalyst for Machine Learning in Fluoroscopy-guided Procedures}},
  date         = {2018},
  booktitle    = {Proc. Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  publisher    = {Springer},
}
```
The IJCARS paper describes the integration of tool modeling and provides quantitative results:
```
@article{DeepDRR2019,
  author       = {Unberath, Mathias and Zaech, Jan-Nico and Gao, Cong and Bier, Bastian and Goldmann, Florian and Lee, Sing Chun and Fotouhi, Javad and Taylor, Russell and Armand, Mehran and Navab, Nassir},
  title        = {{Enabling Machine Learning in X-ray-based Procedures via Realistic Simulation of Image Formation}},
  year         = {2019},
  journal      = {International journal of computer assisted radiology and surgery (IJCARS)},
  publisher    = {Springer},
}
```

## Acknowledgments
CUDA Cubic B-Spline Interpolation (CI) used in the projector:  
https://github.com/DannyRuijters/CubicInterpolationCUDA  
D. Ruijters, B. M. ter Haar Romeny, and P. Suetens. Efficient GPU-Based Texture Interpolation using Uniform B-Splines. Journal of Graphics Tools, vol. 13, no. 4, pp. 61-69, 2008.  

The projector is a heavily modified and ported version of the implementation in CONRAD:  
https://github.com/akmaier/CONRAD  
A. Maier, H. G. Hofmann, M. Berger, P. Fischer, C. Schwemmer, H. Wu, K. Müller, J. Hornegger, J. H. Choi, C. Riess, A. Keil, and R. Fahrig. CONRAD—A software framework for cone-beam imaging in radiology. Medical Physics 40(11):111914-1-8. 2013.  

Spectra are taken from MCGPU:  
A. Badal, A. Badano, Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11): 4878–80.  

The segmentation pipeline is based on the Vnet architecture:  
https://github.com/mattmacy/vnet.pytorch  
F. Milletari, N. Navab, S-A. Ahmadi. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. arXiv:160604797. 2016.

We gratefully acknowledge the support of the NVIDIA Corporation with the donation of the GPUs used for this research.
