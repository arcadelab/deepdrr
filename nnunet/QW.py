from deepdrr import geo, Volume, MobileCArm
from deepdrr.projector import Projector # separate import for CUDA init
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

path = '/home/sean/datasets/DeepDRR_Data/case-100129'
lung = nib.load(path + '/Liver.nii.gz')
lung_arr = np.array(lung.dataobj)
materials = {}
materials["lung"] = lung_arr == 1
np.savez(path + '/materials', materials)

volume = Volume.from_nifti(path + '/case-100129.nii.gz', path + '/materials.npz')
carm = MobileCArm()
carm.reposition(volume.center_in_world)

with Projector(volume, carm=carm) as projector:
    carm.move_to(alpha=30, beta=10, degrees=True)
    projection = projector()

plt.imshow(projection, cmap='gray')
plt.show()

