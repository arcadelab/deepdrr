# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 08:28:03 2022

@author: Ding
"""

import os
import shutil

data_loading_path = '/home/sean/torso_mid_result/data'
docker_shared_path = '/home/sean/cis2/ct_organ_seg_docker/shared'
org_mask_path = '/home/sean/torso_mid_result/org_mask'
file_list = os.listdir(data_loading_path)
#os.system('cd ~/cis2/ct_organ_seg_docker')
existing_file_list = os.listdir(org_mask_path)
for case_name in file_list:
    if case_name not in existing_file_list:
        current_file_path = os.path.join(data_loading_path,case_name)
        shared_file_path = os.path.join(docker_shared_path,case_name)
        shutil.copyfile(current_file_path,shared_file_path)
        
        mask_name = 'seg_'+ case_name
        command_line = 'sh run_docker_container.sh '+ case_name+' '+mask_name
        current_process = os.system(command_line)
        print(current_process)
        shutil.move(os.path.join(docker_shared_path,mask_name),os.path.join(org_mask_path,case_name))
        os.remove(shared_file_path)
        
    
    
    