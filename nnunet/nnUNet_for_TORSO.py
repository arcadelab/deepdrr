# import required module
import glob
import os
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
from shutil import copyfile
# import nnunet

def allrunnnUNet(TaskType):
    # predict use nnUNet pretrained
    # os.system('cmd /k "nnUNet_convert_decathlon_task -i /home/qiyuan/Documents/raw/Task03_Liver"')

    # assign directory
    # dcm_directory = '/home/sean/cis2/data/Pediatric-First-50/Pediatric-CT-SEG'
    # nii_directory = os.environ.get('nnUNet_raw_data_base') + '/nnUNet_raw_data/Pediatric-50/'
    dcm_directory = '/home/sean/cis2/data/torso_volumes'
    # dcm_directory = '/home/qiyuan/Downloads/torso_volumes'

    nii_directory = os.environ.get('nnUNet_raw_data_base') + '/nnUNet_raw_data/TORSO/'

    # Create target Directory if don't exist
    if not os.path.exists(nii_directory):
        os.makedirs(nii_directory)
        os.makedirs(nii_directory + 'imagesTs/')
        os.makedirs(nii_directory + 'imagesTr/')
        os.makedirs(nii_directory + 'labelsTr/')
        os.makedirs(nii_directory + 'labelsTs/')
        print("Directory ", nii_directory, " Created ")

        # iterate over files in
        # that directory
        # i=0
        for filepath in glob.iglob(f'{dcm_directory}/*/*.nii.gz'):
            # i = i+1
            head_tail1 = os.path.split(os.path.normpath(filepath + '/../'))
            head_tail2 = os.path.split(os.path.normpath(filepath))
            if head_tail2[1].find('seg') == -1:
                marker = head_tail1[1] + '_0000.nii.gz'  # This is the name of nii.gz file
                testdatadir = nii_directory + 'imagesTs/'
                copyfile(filepath, join(testdatadir, marker))
            else:
                marker = head_tail1[1] + '_seg.nii.gz'  # This is the name of nii.gz file
                testdatadir = nii_directory + 'labelsTs/'
                copyfile(filepath, join(testdatadir, marker))

            # os.system('/home/sean/cis2/nnUNet/dcm2niix -f "' + marker + '" -p y -z y -o "' + testdatadir + '" "' + filepath +'"')

        # Write into dataset.json
        base = nii_directory
        train_folder = join(base, "imagesTr/")
        label_folder = join(base, "labelsTr/")
        test_folder = join(base, "imagesTs/")
        train_patient_names = []
        test_patient_names = []
        train_patients = subfiles(train_folder, join=False, suffix='nii.gz')
        test_patients = subfiles(test_folder, join=False, suffix=".nii.gz")

        json_dict = OrderedDict()
        json_dict['name'] = "TORSO"
        json_dict['description'] = "test 0320"
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

    else:
        print("Directory ", nii_directory, " already exists")

    os.system('nnUNet_predict -i ' + nii_directory + 'imagesTs/ -o '
              '/home/sean/cis2/nnUNet/nnU_OUTPUT_Task' + str(TaskType) + ' -t ' + str(TaskType) + ' -m 3d_fullres')


if __name__ == "__main__":
    TaskType = 2  # Heart:2, Liver:3, Lung:6
    allrunnnUNet(TaskType)
