<div align="center">

# DeepDRR

<a href="https://arxiv.org/abs/1803.08606">
<img src="http://img.shields.io/badge/paper-arxiv.1803.08606-B31B1B.svg" alt="Paper" />
</a>
<a href="https://pepy.tech/project/deepdrr">
<img src="https://pepy.tech/badge/deepdrr/month" alt="Downloads" />
</a>
<a href="https://github.com/arcadelab/deepdrr/releases/">
<img src="https://img.shields.io/github/release/arcadelab/deepdrr.svg" alt="GitHub release" />
</a>
<a href="https://pypi.org/project/deepdrr/">
<img src="https://img.shields.io/pypi/v/deepdrr" alt="PyPI" />
</a>
<a href="http://deepdrr.readthedocs.io/?badge=latest">
<img src="https://readthedocs.org/projects/deepdrr/badge/?version=latest" alt="Documentation Status" />
</a>
<a href="https://github.com/psf/black">
<img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" />
</a>
<a href="https://colab.research.google.com/github/arcadelab/deepdrr/blob/main/deepdrr_demo.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" />
</a>

</div>

DeepDRR is a GPU-based framework for efficient simulation of X-ray images—or digitally reconstructed radiographs (DRRs)—from 3D CT images and meshes. It is intended for large-scale generation of synthetic X-ray datasets for training machine learning models.

## Usage

The following example loads a CT volume from a NifTi `.nii.gz` file and simulates a single X-ray projection:

```python
from deepdrr import Volume, SimpleDevice, Projector
import killeengeo as kg

device = SimpleDevice()
ct = Volume.from_nifti('/path/to/ct_image.nii.gz')

with Projector(ct, device=device) as projector:
    device.set_view(
        point=ct.center_in_world, 
        direction=ct.world_from_anatomical @ kg.vector(0, 1, 0),
        up=ct.world_from_anatomical @ kg.vector(0, 0, 1), 
        source_to_point_fraction=0.5, 
    )

    image = projector()
```

A more involved example that samples many projections from a CT volume and a tool mesh is below:

```python
from deepdrr import Volume, SimpleDevice, Projector, Mesh
import killeengeo as kg
from PIL import Image

device = SimpleDevice(
    sensor_height=384,
    sensor_width=384,
    pixel_size=0.3, # pixel size in mm
    source_to_detector_distance=1000, # distance from source to detector in mm
)
ct = Volume.from_nifti('/path/to/ct_image.nii.gz')

# Load a tool mesh from an STL file, with homogeneous titanium material, called "tool"
tool = Mesh.from_stl("/path/to/tool_mesh.stl", material="titanium", tag="tool")

# Create a projector that combines the CT volume and the tool mesh
projector = Projector(
    [ct, tool], 
    device=device,
    neglog=False, # do not apply negative log transform
)

right = ct.world_from_anatomical @ kg.vector(1, 0, 0) # right direction in world coordinates
anterior = ct.world_from_anatomical @ kg.vector(0, 1, 0) # anterior direction in world coordinates
superior = ct.world_from_anatomical @ kg.vector(0, 0, 1) # superior direction in world coordinates

# Initialize the projector (allocates GPU memory, takes the most time)
projector.initialize() 

for i in range(100): 
    # Sample a point in world coordinates uniformly *in* a sphere
    point = kg.random.uniform(
        center=ct.center_in_world,
        radius=100, # 100 mm radius around the CT center
    )

    # Sample a direction in world coordinates within a cone around the anterior
    direction = kg.random.spherical_uniform(
        center=anterior,
        d_phi=np.pi / 6, # 30 degrees around the anterior direction
    )

    # Sample how far the source is from the patient
    source_to_point_fraction = np.random.uniform(0.5, 0.9) # Depends on application

    # Randomly sample a pose for the C-arm
    device.set_view( 
        point=point,
        direction=direction,
        up=superior,
        source_to_point_fraction=source_to_point_fraction
    )

    # Place the tool in the center of the CT volume (or any other position)
    tool.place_center(ct.center_in_world)

    # Generate the X-ray image, (float32, not normalized)
    image = projector()

    # Save out the image as a float32 numpy array, with no normalization
    Image.fromarray(image).save(f"path/to/output/{i:04d}.tiff")

    # Get the segmentation of the tool in the image
    seg = projector.project_seg(tags=["tool"]) # [H, W, 1] array with 1 for tool pixels, 0 otherwise

    # Save the segmentation as a binary image
    Image.fromarray(seg.squeeze().astype(np.uint8) * 255).save(f"path/to/output/{i:04d}_seg.png")

projector.free()
```

The script `example_projector.py` gives an alternative example. Additional tutorials are in progress at [deepdrr.readthedocs.io](https://deepdrr.readthedocs.io). Contributions are welcome.

## Installation

Because DeepDRR is a complex package with many dependencies, we recommend installing it from source using [mamba](https://github.com/conda-forge/miniforge) or [conda](https://docs.conda.io/en/latest/). It requires a linux machine with an NVIDIA GPU and CUDA support. 

1. Install the nvidia drivers for your system. [ubuntu guide](https://help.ubuntu.com/community/NvidiaDriversInstallation) 
    - _e.g._ on ubuntu 22, we use `sudo ubuntu-drivers install --gpgpu nvidia:535-server`
2. Install conda (we recommend [mambaforge](https://github.com/conda-forge/miniforge)).
3. Run the following commands:

```bash
git clone https://github.com/arcadelab/deepdrr.git
cd deepdrr
sudo ./scripts/setup_ubuntu.sh # installs apt dependencies and sets up the EGL driver
conda env create -f environment.yml # installs CUDA 12 and pytorch
conda activate deepdrr
pip install .[cuda11x] # this should match your CUDA version, see installation notes below
```

### Installation Notes

#### Other CUDA Versions
DeepDRR depends on cupy which needs a version compiled for your CUDA version. 
When installing DeepDRR, you can specify the CUDA version you want to use by appending the appropriate option to the `pip install` command:
```bash
pip install .[cuda102] # for CUDA 10.2
pip install .[cuda110] # for CUDA 11.0
pip install .[cuda111] # for CUDA 11.1
pip install .[cuda11x] # for CUDA 11.2 - 11.8
pip install .[cuda12x] # for CUDA 12.x
```
See [https://docs.cupy.dev/en/stable/install.html](https://docs.cupy.dev/en/stable/install.html) for more information.

## Installing for Development

Installing from the `dev` branch is risky, as it is unstable.

1. Install the nvidia drivers for your system. [guide](https://help.ubuntu.com/community/NvidiaDriversInstallation) 
2. Install conda (we recommend [mambaforge](https://github.com/conda-forge/miniforge)).
3. Run the following commands:

```bash
git clone https://github.com/arcadelab/deepdrr.git
cd deepdrr
git checkout dev
sudo ./scripts/setup_ubuntu.sh
conda env create -f environment.yml
conda activate deepdrr
pip install -e .[dev,cuda12x] # this should match your CUDA version
```

## Documentation

Documentation is available at [deepdrr.readthedocs.io](https://deepdrr.readthedocs.io/).

To create the autodocs, run

```bash
sphinx-apidoc -f -o docs/source deepdrr
```

in the base directory. Then do `cd docs` and `make html` to build the static site locally.

## Contributing

Contributions for bug fixes, enhancements, and other suggestions are welcome. Please make a pull request.

## Method Overview

DeepDRR combines machine learning models for material decomposition and scatter estimation in 3D and 2D, respectively, with analytic models for projection, attenuation, and noise injection to achieve the required performance. The pipeline is illustrated below.

![DeepDRR Pipeline](https://raw.githubusercontent.com/arcadelab/deepdrr/master/images/deepdrr_workflow.png)

Further details can be found in our MICCAI 2018 paper "DeepDRR: A Catalyst for Machine Learning in Fluoroscopy-guided Procedures" and the subsequent Invited Journal Article in the IJCARS Special Issue of MICCAI "Enabling Machine Learning in X-ray-based Procedures via Realistic Simulation of Image Formation". The conference preprint can be accessed on arXiv here: <https://arxiv.org/abs/1803.08606>.

### Representative Results

The figure below shows representative radiographs generated using DeepDRR from CT data downloaded from the NIH Cancer Imaging Archive. Please find qualitative results in the **Applications** section.

![Representative DeepDRRs](https://raw.githubusercontent.com/arcadelab/deepdrr/master/images/examples.PNG)

### Applications - Pelvis Landmark Detection

We have applied DeepDRR to anatomical landmark detection in pelvic X-ray: "X-ray-transform Invariant Anatomical Landmark Detection for Pelvic Trauma Surgery", also early-accepted at MICCAI'18: <https://arxiv.org/abs/1803.08608> and now with quantitative evaluation in the IJCARS Special Issue on MICCAI'18: <https://link.springer.com/article/10.1007/s11548-019-01975-5>. The ConvNet for prediction was trained on DeepDRRs of 18 CT scans of the NIH Cancer Imaging Archive and then applied to ex vivo data acquired with a Siemens Cios Fusion C-arm machine equipped with a flat panel detector (Siemens Healthineers, Forchheim, Germany). Some representative results on the ex vivo data are shown below.

![Prediction Performance](https://raw.githubusercontent.com/arcadelab/deepdrr/master/images/landmark_performance_real_data.PNG)

### Applications - Metal Tool Insertion

DeepDRR has also been applied to simulate X-rays of the femur during insertion of dexterous manipulaters in orthopedic surgery: "Localizing dexterous surgical tools in X-ray for image-based navigation", which has been accepted at IPCAI'19: <https://arxiv.org/abs/1901.06672>. Simulated images are used to train a concurrent segmentation and localization network for tool detection. We found consistent performance on both synthetic and real X-rays of ex vivo specimens. The tool model, simulation image and detection results are shown below.

This capability has not been tested in version 1.0. For tool insertion, we recommend working with [Version 0.1](https://github.com/arcadelab/deepdrr/releases/tag/0.1) for the time being.

![Robot Insertion and Detection](https://raw.githubusercontent.com/arcadelab/deepdrr/master/images/tool_insertion.png)

### Known Limitations

#### General

1. Our material decomposition V-net was trained on NIH Cancer Imagign Archive data. In case it does not generalize perfectly to other acquisitions, the use of intensity thresholds (as is done in conventional Monte Carlo) is still supported. In this case, however, thresholds will likely need to be selected on a per-dataset, or worse, on a per-region basis since bone density can vary considerably.
2. Scatter estimation is currently limited to Rayleigh scatter and we are working on improving this. Scatter estimation was trained on images with 1240x960 pixels with 0.301 mm. The scatter signal is a composite of Rayleigh, Compton, and multi-path scattering. While all scatter sources produce low frequency signals, Compton and multi-path are more blurred compared to Rayleigh, suggesting that simple scatter reduction techniques may do an acceptable job. In most clinical products, scatter reduction is applied as pre-processing before the image is displayed and accessible. Consequently, the current shortcoming of not providing _full scatter estimation_ is likely not critical for many applications, in fact, scatter can even be turned off completely. We would like to refer to the **Applications** section above for some preliminary evidence supporting this reasoning.
3. GPU memory consumption.
4. We currently provide the X-ray source sprectra from MC-GPU that are fairly standard. Additional spectra can be implemented.
5. The current detector reading is _the average energy deposited by a single photon in a pixel_. We do not support photon counting or energy resolving detectors (yet!).
6. It is important to check proper import of CT volumes. We have tried to account for many variations (HU scale offsets, slice order, origin, file extensions) but one can never be sure enough, so please double check for your files.

#### Mesh Rendering

DeepDRR uses PyRender, which is no longer actively maintained. As a result, there are some known bugs when using meshes in the pipeline.

1. A segfault may occur when the version of PyOpenGL is not compatible or if the version of numpy is 2.0 or higher. To fix this, we require `PyOpenGL==3.1.6` and `numpy<2.0`. 
2. Mesh rendering uses depth peeling to render the mesh, which can lead to errors for very intricate meshes. Meshes are also assumed to be watertight and to **not** contain the source. See the `Projector` class for details.
3. A common error indicates that `EGL` is not available. In other contexts, this indicates that a display is needed, but _not here._ DeepDRR should not require a display to run. This error most likely indicates that the EGL driver is not installed correctly. Please run `sudo ./scripts/setup_ubuntu.sh` to install the EGL driver. 

## Citation

We hope this proves useful for medical imaging research. If you use our work, please consider citing our work.

The 2018 MICCAI article covers the basic DeepDRR pipeline and task-based evaluation:

```bibtex
@inproceedings{DeepDRR2018,
  author       = {Unberath, Mathias and Zaech, Jan-Nico and Lee, Sing Chun and Bier, Bastian and Fotouhi, Javad and Armand, Mehran and Navab, Nassir},
  title        = {{DeepDRR--A Catalyst for Machine Learning in Fluoroscopy-guided Procedures}},
  date         = {2018},
  booktitle    = {Proc. Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  publisher    = {Springer},
}
```

The 2019 IJCARS paper describes the integration of tool modeling and provides quantitative results:

```bibtex
@article{DeepDRR2019,
  author       = {Unberath, Mathias and Zaech, Jan-Nico and Gao, Cong and Bier, Bastian and Goldmann, Florian and Lee, Sing Chun and Fotouhi, Javad and Taylor, Russell and Armand, Mehran and Navab, Nassir},
  title        = {{Enabling Machine Learning in X-ray-based Procedures via Realistic Simulation of Image Formation}},
  year         = {2019},
  journal      = {International journal of computer assisted radiology and surgery (IJCARS)},
  publisher    = {Springer},
}
```

## Version 0.1

For the original DeepDRR, released alongside our 2018 paper, please see the [Version 0.1](https://github.com/arcadelab/deepdrr/releases/tag/0.1).

## Acknowledgments

CUDA Cubic B-Spline Interpolation (CI) used in the projector:  
<https://github.com/DannyRuijters/CubicInterpolationCUDA>  
D. Ruijters, B. M. ter Haar Romeny, and P. Suetens. Efficient GPU-Based Texture Interpolation using Uniform B-Splines. Journal of Graphics Tools, vol. 13, no. 4, pp. 61-69, 2008.

The projector is a heavily modified and ported version of the implementation in CONRAD:  
<https://github.com/akmaier/CONRAD>  
A. Maier, H. G. Hofmann, M. Berger, P. Fischer, C. Schwemmer, H. Wu, K. Müller, J. Hornegger, J. H. Choi, C. Riess, A. Keil, and R. Fahrig. CONRAD—A software framework for cone-beam imaging in radiology. Medical Physics 40(11):111914-1-8. 2013.

Spectra are taken from MCGPU:  
A. Badal, A. Badano, Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel graphics processing unit. Med Phys. 2009 Nov;36(11): 4878–80.

The segmentation pipeline is based on the Vnet architecture:  
<https://github.com/mattmacy/vnet.pytorch>  
F. Milletari, N. Navab, S-A. Ahmadi. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. arXiv:160604797. 2016.

We gratefully acknowledge the support of the NVIDIA Corporation with the donation of the GPUs used for this research.
