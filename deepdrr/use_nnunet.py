import logging
import numpy as np
from pathlib import Path
import torch
from torch.autograd import Variable

import nnunet
# from .network_segmentation import VNet
from .utils import data_utils

logger = logging.getLogger(__name__)
# import required module
import glob
import os
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *

class Segmentation():
    def __init__(self):
        temp_dir = '/nnUNet_raw_data/temp/'  #os.environ.get('nnUNet_raw_data_base') + 
        
    def dataprep(self,idir,type):
        # assign directory
    #     out_directory = os.environ.get('nnUNet_raw_data_base') + '/nnUNet_raw_data/temp/'
        out_directory = self.temp_dir
        # Create target Directory if don't exist
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)
            os.makedirs(out_directory + 'imagesTs/')
            os.makedirs(out_directory + 'imagesTr/')
            os.makedirs(out_directory + 'labelsTr/')
            print("Directory ", out_directory, " Created ")
        else:
            print("Directory ", out_directory, " Already Exists ")
    
        if type == 'nii':
            copyfile(idir, join(out_directory + 'imagesTs/', 'temp_0000.nii.gz'))
        else:
            raise NotImplementedError("TODO")
    
        # Write into dataset.json
        base = out_directory
        train_folder = join(base, "imagesTr/")
        label_folder = join(base, "labelsTr/")
        test_folder = join(base, "imagesTs/")
        train_patient_names = []
        test_patient_names = []
        train_patients = subfiles(train_folder, join=False, suffix='nii.gz')
        test_patients = subfiles(test_folder, join=False, suffix=".nii.gz")

        json_dict = OrderedDict()
        json_dict['name'] = "temp"
        json_dict['description'] = "-"
        json_dict['tensorImageSize'] = "3D"
        json_dict['reference'] = "-"
        json_dict['licence'] = "-"
        json_dict['release'] = "-"
        json_dict['modality'] = {
            "0": "CT",
        }
        json_dict['labels'] = OrderedDict({
        }
        )
        json_dict['numTraining'] = len(train_patient_names)
        json_dict['numTest'] = len(test_patients)
        json_dict['training'] = [{'image': "./imagesTr/%s" % train_patients, "label": "./labelsTr/%s" % train_patients} for i, train_patients in enumerate(train_patient_names)]
        json_dict['test'] = ["./imagesTs/%s" % test_patients for test_patients in test_patients]

        save_json(json_dict, os.path.join(base, "dataset.json"))
    
    def infer(self,TaskType):
        nnUNet_download_pretrained_model Task029_LiTS
        nnUNet_predict -i + self.temp_dir + 'imagesTs/ -o '
                  'Task_' + str(TaskType) + ' -t ' + str(TaskType) + ' -m 3d_fullres')
# 1. setup nnunet paths (input / output) (*system path)
# 2. check corresponding pretrained model and download if not exist
# 3. run inference and save mask
# 4. read mask, give value to materials, *delete mask

class Segmentation():
    url = "https://www.dropbox.com/s/pn4aw4z2i01eoo4/model_segmentation.pth.tar?dl=1"
    # md5 = "73201847d381131f7e6753e40252dfbc"

    filename = "model_segmentation.pth.tar"

    def __init__(self):
        torch.cuda.set_device(0)

        self.model_path = self.download()
        self.model = VNet()
        self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(self.model_path)['state_dict'])
        self.model.eval()
        logger.info("loaded segmentation network")

    def download(self) -> Path:
        return data_utils.download(self.url, self.filename)

    def segment(self, input_volume, show_results=False):
        segmentation_prior_air = 1
        segmentation_prior_soft = 1
        segmentation_prior_bone = 1

        mean = -630.1
        std = 479.7
        blocksize = 128

        sanity_check_volume = np.zeros(
            [3, input_volume.shape[0], input_volume.shape[1], input_volume.shape[2]])
        sanity_check_volume[0, :, :, :] = input_volume <= -700
        sanity_check_volume[1, :, :, :] = (
            input_volume <= 400) * (input_volume > -800)
        sanity_check_volume[2, :, :, :] = input_volume > 150

        blocks = np.ceil(np.array(input_volume.shape) / blocksize).astype(int)
        new_shape = blocks * blocksize
        offset_before = ((new_shape - input_volume.shape) // 2).astype(int)
        offset_after = ((new_shape - input_volume.shape) -
                        offset_before).astype(int)
        padded_volume = np.pad(input_volume, [[offset_before[0], offset_after[0]], [
                               offset_before[1], offset_after[1]], [offset_before[2], offset_after[2]]], mode='edge')

        padded_volume -= mean
        padded_volume /= std

        segmented_volume = np.zeros(
            [3, padded_volume.shape[0], padded_volume.shape[1], padded_volume.shape[2]], dtype=np.float32)
        logger.debug(segmented_volume.shape)
        counter = 1
        for i in range(0, blocks[0]):
            for j in range(0, blocks[1]):
                for k in range(0, blocks[2]):
                    logger.debug(
                        f'segmenting block {counter} / {blocks[0] * blocks[1] * blocks[2]}')
                    counter += 1
                    curren_block = padded_volume[i * blocksize:(i + 1) * blocksize, j * blocksize:(
                        j + 1) * blocksize, k * blocksize:(k + 1) * blocksize]
                    presegmentation = np.zeros(
                        (4, blocksize, blocksize, blocksize), dtype=np.float32)
                    presegmentation[0, :, :, :] = curren_block
                    presegmentation[1, :, :, :] = curren_block >= (
                        200 - mean) / std
                    presegmentation[2, :, :, :] = (curren_block < (
                        200 - mean) / std) * (curren_block >= (-500 - mean) / std)
                    presegmentation[3, :, :,
                                    :] = curren_block < (-500 - mean) / std
                    curren_block_tensor = torch.from_numpy(
                        presegmentation).cuda()
                    curren_block_tensor = torch.unsqueeze(
                        curren_block_tensor, 0)
                    output_tensor = self.model.forward(
                        Variable(curren_block_tensor, requires_grad=False)).cpu().detach().numpy()
                    segmented_volume[:, i * blocksize:(i + 1) * blocksize, j * blocksize:(
                        j + 1) * blocksize, k * blocksize:(k + 1) * blocksize] = output_tensor[0, :, :, :, :]
        segmented_volume = segmented_volume[:, offset_before[0]:input_volume.shape[0] + offset_before[0], offset_before[1]                                            :input_volume.shape[1] + offset_before[1], offset_before[2]:input_volume.shape[2] + offset_before[2]]

        # ensure correct label
        segmented_volume *= sanity_check_volume
        segmented_volume[0, :, :, :] *= segmentation_prior_air
        segmented_volume[1, :, :, :] *= segmentation_prior_soft
        segmented_volume[2, :, :, :] *= segmentation_prior_bone
        segmented_volume = np.argmax(segmented_volume, axis=0)
        segmentation = {}

        # Air
        segmentation["air"] = segmented_volume == 0

        # Soft Tissue
        segmentation["soft tissue"] = segmented_volume == 1

        # Bone
        segmentation["bone"] = segmented_volume == 2

        return segmentation
