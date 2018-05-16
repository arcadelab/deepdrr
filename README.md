# DeepDRR
Implementation of our early-accepted MICCAI'18 paper "DeepDRR: A Catalyst for Machine Learning in Fluoroscopy-guided Procedures". 
The paper can be accessed on arXiv here:  https://arxiv.org/abs/1803.08606.

Implemented in Python, PyCuda, and PyTorch.

### Introduction

DeepDRR aims at providing medical image computing and computer assisted intervention researchers state-of-the-art tools to generate realistic radiographs and fluoroscopy from 3D CTs on a trainingset scale. 


### Method Overview
To this end, DeepDRR combines machine learning models for material decomposition and scatter estimation in 3D and 2D, respectively, with analytic models for projection, attenuation, and noise injection to achieve the required performance. The pipeline is illustrated below. 

![DeepDRR Pipeline](https://raw.githubusercontent.com/mathiasunberath/DeepDRR/master/readme_images/deepdrr_workflow.PNG)

### Representative Results
The figure below shows representative radiographs generated using DeepDRR from CT data downloaded from the NIH Cancer Imaging Archive. Please find some indirect results in the **Applications** section.

![Representative DeepDRRs](https://raw.githubusercontent.com/mathiasunberath/DeepDRR/master/readme_images/examples.PNG)

### Applications

We have applied DeepDRR to anatomical landmark detection in pelvic X-ray: "X-ray-transform Invariant Anatomical Landmark Detection for Pelvic Trauma Surgery", also early-accepted at MICCAI'18: https://arxiv.org/abs/1803.08608. The ConvNet for prediction was trained on DeepDRRs of 18 CT scans of the NIH Cancer Imaging Archive and then applied to ex vivo data acquired with a Siemens Cios Fusion C-arm machine equipped with a flat panel detector (Siemens Healthineers, Forchheim, Germany). Some representative results are shown below.

![Prediction Performance](https://raw.githubusercontent.com/mathiasunberath/DeepDRR/master/readme_images/landmark_performance_real_data.PNG)

### Potential Challenges

1. Our material decomposition V-net was trained on NIH Cancer Imagign Archive data. In case it does not generalize perfectly to other acquisitions, the use of intensity thresholds (as is done in conventional Monte Carlo) is still supported. In this case, however, thresholds will likely need to be selected on a per-dataset, or worse, on a per-region basis since bone density can vary considerably.
2. Scatter estimation is currently limited to Rayleigh scatter and we are working on improving this. Scatter estimation was trained on images with 1240x960 pixels with 0.301 mm. The scatter signal is a composite of Rayleigh, Compton, and multi-path scattering. While all scatter sources produce low frequency signals, Compton and multi-path are more blurred compared to Rayleigh, suggesting that simple scatter reduction techniques may do an acceptable job. In most clinical products, scatter reduction is applied as pre-processing before the image is displayed and accessible. Consequently, the current shortcoming of not providing *full scatter estimation* is likely not critical for many applications, in fact, scatter can even be turned off completely. We would like to refer to the **Applications** section above for some preliminary evidence supporting this reasoning.
3. Due to the nature of volumetric image processing, DeepDRR consumes a lot of GPU memory. We have successfully tested on 12 GB of GPU memory but cannot tell about 8 GB at the moment. The bottleneck is volumetric segmentation, which can be turned off and replaced by thresholds (see 1.).
4. We currently provide the X-ray source sprectra from MC-GPU that are fairly standard. Additional spectra can be implemented in spectrum_generator.py. 
5. The current detector reading is *the average energy deposited by a single photon in a pixel*. If you are interested in modeling photon counting or energy resolving detectors, then you may want to take a look at mass_attenuation(_gpu).py to implement your detector.
6. Currently we do not support import of full projection matrices. But you will need to define K, R, and T seperately or use camera.py to define projection geometry online. 
7. It is important to check proper import of CT volumes. We have tried to account for many variations (HU scale offsets, slice order, origin, file extensions) but one can never be sure enough, so please double check for your files. 

## Reference

We hope this proves useful for medical imaging research. If you use our work, we would kindly ask you to reference our MICCAI article:
```
@inproceedings{DeepDRR2018,
  author       = {Unberath, Mathias and Zaech, Jan-Nico and Lee, Sing Chun and Bier, Bastian and Fotouhi, Javad and Armand, Mehran and Navab, Nassir},
  title        = {{DeepDRR--A Catalyst for Machine Learning in Fluoroscopy-guided Procedures}},
  date         = {2018},
  booktitle    = {Proc. Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  publisher    = {Springer},
}
```


### Instructions for Windows:

**Install CUDA 8.0**
1. ```conda create -n pytorch python=3.6```
2. ```conda activate pytorch```

**Install packages**
1. Numpy+MKL from https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
2. ```conda install matplotlib```
3. ```conda install -c conda-forge numexpr```
4. ```conda install -c conda-forge pydicom```
5. ```conda install -c anaconda scikit-image```
6. ```pip install pycuda```
7. ```Pip install tensorboard```
8. ```Pip install tensorboardX```

**Install pytorch**
1. Follow [peterjc123's scripts to run PyTorch on Windows](https://github.com/peterjc123/pytorch-scripts "peterjc123 PyTorch").
2. ```conda install -c peterjc123 pytorch```
3. ```pip install torchvision```
  
**PyCuda not working?**
* Try to add C compiler to path. Most likely the path is: “C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\”.

