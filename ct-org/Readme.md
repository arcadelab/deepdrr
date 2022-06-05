# Instruction for CT-ORG based mask generation

CT-ORG is a 5-classes abdominal organ segmentation model as in [CT-ORG, a new dataset for multiple organ segmentation in computed tomography](https://www.nature.com/articles/s41597-020-00715-8). And the trained network is packaged in docker in [this link](https://github.com/bbrister/ct_organ_seg_docker).

## Mask generation steps
1. Installing the pre-trained models from [previous link](https://github.com/bbrister/ct_organ_seg_docker).
2. Running org_mask_batch.py.(Currently I/O part still in in file for modification, before publication this should be worked in command line format). 
3. In some cases the original docker setting up will not work due to the following reasons:
   - For issues with CPU/GPUs, add `--gpus gpu_index` in ./docker/run_docker_container.py line `sudo docker run --gpus all -v $HOST_SHARED:$CONTAINER_SHARED -t $IMAGE $INFILE $OUTFILE` 
   - For issues with `IOError: CRC check failed`, this is due to `nibabel` or nii data version, change `get_data` to `get_fdata`.
