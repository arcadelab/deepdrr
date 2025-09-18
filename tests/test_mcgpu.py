import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils
import numpy as np
import logging
import deepdrr.projector.conv_to_mcgpu as conv_to_mcgpu
import os, subprocess
from PIL import Image

from pathlib import Path

log = logging.getLogger(__name__)

# NOTE on requirements for running this test file:
# This test file assumes that MCGPU is installed on the machine in the following way:
#
#   some-directory
#       |
#       |___ deepdrr
#       |       |
#       |       |___ deepdrr
#       |       |       |
#       |       |       |___ projector
#       |       |       |
#       |       |       |___ geo
#       |       |       |
#       |       |      ...
#       |       |
#       |       |___ examples
#       |       |
#       |       |___ tests
#       |       |
#       |      ...
#       |
#       |___  MCGPU
#       |       |
#      ...      |___ MC-GPU_material_files
#               |       |
#               |       |___ adipose_ICRP110__5-120keV.mcgpu.gz
#               |       |
#               |       |___ air_5-120keV.mcgpu.gz
#               |       |
#               |      ...
#               |
#               |___ 90kVp_4.0mmAl.spc
#               |
#               |___ MC-GPU_v1.3.x
#               |
#              ...
#
#

def test_mcgpu():
    file_path = test_utils.download_sampledata("CT-chest")
    volume = deepdrr.Volume.from_nrrd(file_path)

    # Steps in the test
    #
    # 1. check that MCGPU is installed in correct location. If not, automatically pass the test
    # 2. load volume and specify test's geometry
    # 3. Convert to MCGPU format and write out to file in proper location
    # 4. Run DeepDRR
    # 5. Run MCGPU
    # 6. Compare in intelligent ways

    # 1. check that MCGPU is installed in correct location. If not, automatically pass the test
    tests_dir = Path(__file__).resolve().parent # deepdrr/tests
    ancestor = tests_dir.parent.parent
    mcgpu_dir = ancestor / "MCGPU"
    if not mcgpu_dir.is_dir():
        log.info(f"MCGPU not installed in proper location for {__file__}")
        return
    mcgpu_exe = mcgpu_dir / "MC-GPU_v1.3.x"
    if not mcgpu_exe.is_file():
        log.info(f"MC-GPU_v1.3.x file not installed in proper location for {__file__}")
        return

    mcgpu_test_dir = mcgpu_dir / "DeepDRR_testing"
    if not mcgpu_test_dir.exists():
        os.mkdir(mcgpu_test_dir)

    # 2. load volume and specify test's geometry
    carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
    projector = deepdrr.Projector(
        volume=volume,
        carm=carm,
        step=0.1,
        mode="linear",
        max_block_index=200,
        spectrum="90KV_AL40",
        photon_count=100000,
        scatter_num=10e7,
        threads=8,
        neglog=False,
    )

    # 3. Convert to MCGPU format and write out to file in proper location
    source_world_cm = carm.get_camera_projection().get_center_in_world() * 0.1 # convert to [cm]

    _detector_ctr = np.array([carm.sensor_width / 2, carm.sensor_height / 2, 1])
    src_dir = np.array(carm.get_camera_projection().world_from_index) @ _detector_ctr
    mag2 = (src_dir[0] * src_dir[0]) + (src_dir[1] * src_dir[1]) + (src_dir[2] * src_dir[2])
    source_direction = src_dir / np.sqrt(mag2)

    detector_shape = (carm.sensor_width, carm.sensor_height)

    detector_size_cm = (
        carm.sensor_width * carm.pixel_size * 0.1, # convert to [cm]
        carm.sensor_height * carm.pixel_size * 0.1 # convert to [cm]
    )

    src_to_iso_cm = (np.array(carm.isocenter) * 0.1) - np.array(source_world_cm)
    source_to_isocenter_dist_cm = np.sqrt(np.dot(src_to_iso_cm, src_to_iso_cm))

    FILENAME = "DeepDRR_test_mcgpu"

    log.info("Starting conversion to MCGPU format")

    conv_to_mcgpu.make_mcgpu_inputs(
        volume,
        FILENAME,
        mcgpu_test_dir,
        projector.scatter_num,
        12345,
        projector.threads * projector.threads,
        projector.histories_per_thread,
        "90KV_AL40",
        source_world_cm,
        source_direction,
        detector_shape,
        detector_size_cm,
        carm.source_to_detector_distance * 0.1, # convert to [cm]
        source_to_isocenter_dist_cm
    )

    log.info("Done with conversion to MCGPU format")

    # 4. Run DeepDRR
    projector.initialize()
    deepdrr_image = projector.project()
    #deepdrr_image = (deepdrr_image * 255).astype(np.uint8) # only if using neglog
    
    try:
        d = Path(__file__).resolve().parent
        truth = d / "reference"
        output_dir = d / "output"
        output_dir.mkdir(exist_ok=True)
        Image.fromarray(image).save(output_dir / "test_mcgpu_deepdrr.png")
    except e:
        print(e)
        
    projector.free()
    
    output_dir = test_utils.get_output_dir()
    
    # 5. Run MCGPU
    mcgpu_infile = mcgpu_test_dir / f"{FILENAME}.in"
    mcgpu_outfile = mcgpu_test_dir / f"{FILENAME}.out"
    args = [
        mcgpu_exe.as_posix(), 
        mcgpu_infile.as_posix(), 
        "|", "tee", 
        mcgpu_outfile.as_posix()
    ]
    print("args to subprocess call:")
    for arg in args:
        print(arg)
    #subprocess.call(args) # TODO: figure out how to use this

    cwd_str = os.getcwd()
    print(f"TEMP: current working directory before call: {cwd_str}")
    os.chdir(f"{mcgpu_test_dir.as_posix()}")
    print(f"TEMP: changed directory to {os.getcwd()}")
    os.system(f"{mcgpu_exe.as_posix()} {mcgpu_infile.as_posix()} | tee {mcgpu_outfile.as_posix()}")
    os.chdir(f"{cwd_str}")
    print(f"TEMP: current working directory after call: {os.getcwd()}")

    # 6. Compare in intelligent ways
    dat_file_path = mcgpu_test_dir / f"mcgpu_image_{FILENAME}.dat"
    mcgpu_results_raw = mcgpu_raw_to_ndarray_helper(
            dat_file_path.as_posix(),
            carm.sensor_width,
            carm.sensor_height
        )
    # Convert from [eV/cm^2 per history] to [eV] on each pixel
    pixel_area_cm = (carm.pixel_size * 0.1) * (carm.pixel_size * 0.1)
    mcgpu_results_eV = mcgpu_results_raw * pixel_area * projector.scatter_num

    # TODO continue with comparisons


def mcgpu_raw_to_ndarray_helper(
    absolute_path: str,
    detector_width: int, # MCGPU 'X' direction
    detector_height: int # MCGPU 'Z' direction
) -> np.ndarray:
    """Compile the selected columns from the MCGPU output .dat file into an ndarray with
    dimensions (H, W, 4).

    The returned ndarray is arranged as follows:

                'X' dir -->
       'Z'  [[ _, _, _, ..., _ ], 
             [ _, _, _, ..., _ ],
        |    ... ...        ...
        v    ...    ...     ...
             ...        ... ...
             [ _, _, _, ..., _ ]]
    
    where each 'pixel' contains the array:

        [ UNSCATTERED, COMPTON, RAYLEIGH, MULTI-SCATTERED ]
    
    Args:
        absolute_path (str): the absolute path of the .dat file to read from
        detector_width (int): the number of pixels in the width of the detector (MCGPU 'X' direction)
        detector_height (int): the number of pixels in the height of the detector (MCGPU 'Z' direction)
    """
    ret = np.zeros((detector_height, detector_width)).astype(np.float32)
    
    if not (select_unscattered or select_compton or select_rayleigh or select_multi):
        return ret

    dat_file = Path(absolute_path)
    if not dat_file.is_file():
        log.exception(f"MCGPU .dat file \"{absolute_path}\" does not exist")
        return ret

    dat_file = open(absolute_path, "r")

    UNSCATTERED_COL = 0
    COMPTON_COL = 1
    RAYLEIGH_COL = 2
    MULTI_COL = 3

    row_idx = 0
    col_idx = 0
    line_idx = 0
    
    for line in dat_file:
        line_idx += 1
        if (line[0] is '#') or (line[0] is '\n'):
            continue
        line = line.strip()
        columns = line.split()
        if len(columns) is not 4:
            log.exception(f"line #{line_idx} in MCGPU .dat file \"{absolute_path}\" does not have 4 columns")
            return ret
        for entry in range(4):
            ret[row_idx, col_idx, entry] += float(columns[entry])

        col_idx += 1
        if col_idx > detector_width:
            col_idx = 0
            row_idx += 1
    # end for-loop over lines        
    

if __name__ == "__main__":
    logging.getLogger("deepdrr").setLevel(logging.DEBUG)
    test_mcgpu()
