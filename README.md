# DeepDRR
Implementation of our early-accepted MICCAI'18 paper "DeepDRR: A Catalyst for Machine Learning in Fluoroscopy-guided Procedures". 
The paper can be accessed on arXiv here:  https://arxiv.org/abs/1803.08606

DeepDRR applied to anatomical landmark detection in pelvic X-ray: "X-ray-transform Invariant Anatomical Landmark Detection for Pelvic Trauma Surgery", also early-accept at MICCAI'18: https://arxiv.org/abs/1803.08608

![Prediction Performance](https://raw.githubusercontent.com/mathiasunberath/DeepDRR/master/readme_images/landmark_performance_real_data.PNG)

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

