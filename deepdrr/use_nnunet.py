import logging
import numpy as np
from pathlib import Path
import torch
from torch.autograd import Variable

# import nnunet
# from .network_segmentation import VNet
# from .utils import data_utils

logger = logging.getLogger(__name__)
# import required module
import glob
import os
import subprocess
from collections import OrderedDict
import shutil
from batchgenerators.utilities.file_and_folder_operations import *
import nibabel as nib

class Segmentation():
    
    temp_dir = 'temp/'
        
    def __init__(self):
        temp_dir = 'temp/'  #os.environ.get('nnUNet_raw_data_base') + 
        
    def dataprep(self,idir,type='nii'):
        # assign directory
    #     out_directory = os.environ.get('nnUNet_raw_data_base') + '/nnUNet_raw_data/temp/'
        out_directory = self.temp_dir
        print(out_directory)
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
            shutil.copyfile(idir, os.path.join(out_directory + 'imagesTs/', 'temp_0000.nii.gz'))
        else:
            raise NotImplementedError("TODO")
    
        # Write into dataset.json
        base = out_directory
        train_folder = os.path.join(base, "imagesTr/")
        label_folder = os.path.join(base, "labelsTr/")
        test_folder = os.path.join(base, "imagesTs/")
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
    
    def infer(self, TaskType=17):
        task_name = {
        17: "Task017_AbdominalOrganSegmentation", 
        6: "Task006_Lung"
        }

#         subprocess.call('nnUNet_download_pretrained_model ' + task_name[TaskType], shell = True)
#         subprocess.call('nnUNet_predict -i ' + self.temp_dir + 'imagesTs/ -o ' + self.temp_dir +
#               'Task_' + str(TaskType) + ' -t ' + str(TaskType) + ' -m 3d_fullres', shell = True)
        subprocess.call(['nnUNet_download_pretrained_model', task_name[TaskType]])
        subprocess.call(['nnUNet_predict', '-i', self.temp_dir + 'imagesTs/', '-o', self.temp_dir +
              'Task_' + str(TaskType), '-t', str(TaskType), '-m', '3d_fullres'])
    
    def segment(self, segmented_volume, TaskType=17):
        
        segmentation = {}
        
        if TaskType==0:
            # Air
            segmentation["air"] = segmented_volume == 1
            
            # Bone
            segmentation["bone"] = segmented_volume == 2
            
            #Soft Tissue
            segmentation["soft tissue"] = segmented_volume > 2
        
        if TaskType==6:
            # Soft Tissue
            segmentation["soft tissue"] = segmented_volume == 0
            
            # Lung
            segmentation["Spleen"] = segmented_volume > 0
        
        if TaskType==17:
            segmentation = {}

            # Soft Tissue
            segmentation["soft tissue"] = segmented_volume < 14
            
            # Spleen
            segmentation["Spleen"] = segmented_volume == 1

            # Liver
            segmentation["Liver"] = segmented_volume == 6

        return segmentation
    
    def clear_temp(self):
        os.rmdir(self.temp_dir)
        
    def nnu_segmentation(self, input, TaskType=17):
        self.dataprep(input)
        self.infer(TaskType)
        seg_volume = nib.load(self.temp_dir + 'Task_' + str(TaskType))
        seg_volume_arr=seg_volume.get_fdata()
        segmentation = self.segment(seg_volume_arr, TaskType)
        self.clear_temp()
        return segmentation
    
    def read_mask(self, dir, LabelType=0):
        seg_volume = nib.load(dir)
        seg_volume_arr=seg_volume.get_fdata()
        segmentation = self.segment(seg_volume_arr, LabelType)
        return segmentation
    
# 1. setup nnunet paths (input / output) (*system path)
# 2. check corresponding pretrained model and download if not exist
# 3. run inference and save mask
# 4. read mask, give value to materials, *delete mask
