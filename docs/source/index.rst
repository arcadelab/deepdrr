.. DeepDRR documentation master file, created by
   sphinx-quickstart on Wed Jan 13 15:26:19 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DeepDRR
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

DeepDRR provides state-of-the-art tools to generate realistic 
radiographs and fluoroscopy from 3D CTs on a training set scale.
It is straightforward and user-friendly.

.. code-block:: python
   :linenos:
   import deepdrr
   
   volume = deepdrr.Volume.from_nifti('/path/to/ct_image.nii.gz')
   camera_intrinsics = deepdrr.geo.CameraIntrinsicTransform.from_sizes(
      sensor_size=512,
      pixel_size=0.33,
      source_to_detector_distance=1200,
   )


   center = volume.world_from_ijk @ deepdrr.geo.point(100, 100, 100)
   carm = deepdrr.CArm(isocenter=center)
   
   with deepdrr.Projector(volume, camera_intrinsics, carm) as projector:
      projection = projector()


Installation
------------

DeepDRR requires an NVIDIA GPU, preferably with >11 GB of memory.

1. Install CUDA. Version 11 is recommended, but DeepDRR has been used with 8.0
2. Make sure your C compiler is on the path. DeepDRR has been used with `gcc 9.3.0`.
3. Install from `PyPI`

   pip install deepdrr

Contribute
----------

Contributions are welcome! Please make a pull request.

- Issue Tracker: github.com/arcadelab/deepdrr/issues
- Source Code: github.com/arcadelab/deepdrr

License
-------

DeepDRR is licensed under the GPL-3.0 License.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
