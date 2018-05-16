# DeepDRR
Implementation of our early-accepted MICCAI'18 paper "DeepDRR: A Catalyst for Machine Learning in Fluoroscopy-guided Procedures". 
The paper can be accessed on arXiv here:  https://arxiv.org/abs/1803.08606

## Introduction

DeepDRR aims at providing medical image computing and computer assisted intervention researchers state-of-the-art tools to generate realistic radiographs and fluoroscopy from 3D CTs on a trainingset scale. To this end, DeepDRR combines machine learning models for material decomposition and scatter estimation in 3D and 2D, respectively, with analytic models for projection, attenuation, and noise injection to achieve the required performance. The pipeline is illustrated below. 

![DeepDRR Pipeline](https://raw.githubusercontent.com/mathiasunberath/DeepDRR/master/readme_images/deepdrr_workflow.PNG)

We have applied DeepDRR to anatomical landmark detection in pelvic X-ray: "X-ray-transform Invariant Anatomical Landmark Detection for Pelvic Trauma Surgery", also early-accepted at MICCAI'18: https://arxiv.org/abs/1803.08608. The ConvNet for prediction was trained on DeepDRRs of 18 CT scans of the NIH Cancer Imaging Archive and then applied to ex vivo data acquired with a Siemens Cios Fusion C-arm machine equipped with a flat panel detector (Siemens Healthineers, Forchheim, Germany). Some representative results are shown below.

![Prediction Performance](https://raw.githubusercontent.com/mathiasunberath/DeepDRR/master/readme_images/landmark_performance_real_data.PNG)

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


## Instructions for Windows:

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

