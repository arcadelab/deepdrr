# import required module
import glob
import os
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
# import nnunet

def dataprep(idir,fname,type):
    # predict use nnUNet pretrained
    # os.system('cmd /k "nnUNet_convert_decathlon_task -i /home/qiyuan/Documents/raw/Task03_Liver"')
    # assign directory
    out_directory = os.environ.get('nnUNet_raw_data_base') + '/nnUNet_raw_data/' + fname + '/'
    # Create target Directory if don't exist
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
        os.makedirs(out_directory + 'imagesTs/')
        os.makedirs(out_directory + 'imagesTr/')
        os.makedirs(out_directory + 'labelsTr/')
        print("Directory ", out_directory, " Created ")

        # iterate over files in
        # that directory
        if type == 'ped':
            i=0
            for filepath in glob.iglob(f'{idir}/*/*/*-CT-*'):  #specially for pediatric dataset
                i = i+1
                head_tail = os.path.split(os.path.normpath(filepath + '/../../'))
                marker = str(i) + '-' + head_tail[1] + '_0000'   #This is the name of nii.gz file
                testdatadir = out_directory + 'imagesTs/'
                os.system('/home/sean/cis2/nnUNet/dcm2niix -f "' + marker + '" -p y -z y -o "' + testdatadir + '" "' + filepath +'"')

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
        json_dict['name'] = "Pediatric-50"
        json_dict['description'] = "test 0309"
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
        print("Directory ", out_directory, " already exists, nothing to be done.")


if __name__ == "__main__":
    input = '/home/sean/zding20/pediatric_dataset/exp_debug3'
    foldername = '0403'
    dataprep(input,foldername,'ped')

