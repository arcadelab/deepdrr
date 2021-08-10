#
# Helper file to generate MCGPU inputs from a DeepDRR volume
#
from typing import Tuple
import logging
import numpy as np

from .. import vol

from pathlib import Path

log = logging.getLogger(__name__)


def get_mat_filename(deepDRR_mat_name: str) -> str:
    """Material names are those from the dictionary in material_coefficients.py file
    """
    if "bone" == deepDRR_mat_name:
        return "bone_ICRP110"
    if "soft tissue" == deepDRR_mat_name:
        return "soft_tissue_ICRP110"
    if "air" == deepDRR_mat_name:
        return "air"
    if "iron" == deepDRR_mat_name:
        log.exception(f"UNSUPPORTED MATERIAL FOR MCGPU: {deepDRR_mat_name}")
        return "UNSUPPORTED_MATERIAL"
    if "lung" == deepDRR_mat_name:
        return "lung_ICRP110"
    if "titanium" == deepDRR_mat_name:
        return "titanium"
    if "teflon" == deepDRR_mat_name:
        log.exception(f"UNSUPPORTED MATERIAL FOR MCGPU: {deepDRR_mat_name}")
        return "UNSUPPORTED_MATERIAL"
    log.exception(f"INVALID MATERIAL NAME: {deepDRR_mat_name}")
    return "INVALID_MATERIAL_NAME"


def make_mcgpu_inputs(
    geom: vol.Volume,
    filename: str,
    target_dir: Path, # is a subdirectory in the MCGPU directory
    histories: int,
    seed: int,
    threads_per_block: int,
    histories_per_thread: int,
    spectrum: str,
    source_xyz_cm: np.ndarray,
    source_direction: np.ndarray,
    detector_pixels: Tuple[int, int],
    detector_size_cm: Tuple[float, float],
    source_to_detector_distance_cm: float,
    source_to_isocenter_distance_cm: float,
) -> None:
    """Creates multiple files to serve as the inputs
    """

    # Create the material-ID relation

    mat_mapping = {}  # name-to-ID
    idx = 1
    for mat_name in geom.materials:
        mat_mapping[mat_name] = idx
        idx += 1

    assert idx == (1 + len(geom.materials))

    id_mapping = {}  # ID-to-name
    for mat_id in range(1, idx):
        for mat_name in mat_mapping:
            if mat_mapping[mat_name] == mat_id:
                id_mapping[mat_id] = mat_name
                break

    for name in mat_mapping:
        assert id_mapping[mat_mapping[name]] == name
    for mat_id in range(1, idx):
        assert mat_mapping[id_mapping[mat_id]] == mat_id

    voxel_file = open(f"{target_dir.as_posix()}/{filename}.vox", "w")

    voxel_file.write(f"[SECTION VOXELS HEADER v.2008-04-13]\n")
    voxel_file.write(
        f" {geom.shape[0]} {geom.shape[1]} {geom.shape[2]} No. OF VOXELS IN X,Y,Z\n"
    )

    cm_spacing = 0.1 * np.array(geom.spacing)
    voxel_file.write(
        f" {cm_spacing[0]} {cm_spacing[1]} {cm_spacing[2]} VOXEL SIZE (cm) ALONG X,Y,Z\n"
    )

    voxel_file.write(f"1 COLUMN NUMBER WHERE MATERIAL ID IS LOCATED\n")
    voxel_file.write(f"2 COLUMN NUMBER WHERE THE MASS DENSITY IS LOCATED\n")
    voxel_file.write(f"1 BLANK LINES AT END OF X,Y-CYCLES (1=YES,0=NO)\n")
    voxel_file.write(f"[END OF VXH SECTION]\n")

    for z in range(geom.shape[2]):
        for y in range(geom.shape[1]):
            for x in range(geom.shape[0]):
                for mat_name in geom.materials:
                    if geom.materials[mat_name][x, y, z]:
                        mat_id = mat_mapping[mat_name]
                        density = geom.data[x, y, z]
                        if density < 0:
                            log.exception(
                                f"density in volume voxel ({x}, {y}, {z}) is negative: {density}"
                            )
                            voxel_file.close()
                            return
                        elif density == 0:
                            # MCGPU does not allow for zero density
                            density = 10e-6
                        voxel_file.write(f"{mat_id} {density}\n")
                        break
                # end loop through materials
            voxel_file.write("\n")
        voxel_file.write("\n")

    voxel_file.close()

    mcgpu_infile = open(f"{target_dir.as_posix()}/{filename}.in", "w")

    mcgpu_infile.write(f"#[SECTION SIMULATION CONFIG v.2009-05-12]\n")

    tmp = f"{histories:.2E}".replace("+", "")
    mcgpu_infile.write(
        f"{tmp} # TOTAL NUMBER OF HISTORIES, OR SIMULATION TIME IF VALUE < 10^5\n"
    )
    mcgpu_infile.write(f"{seed} # RANDOM SEED (ranecu PRNG)\n")
    mcgpu_infile.write(
        f"0 # GPU NUMBER TO USE WHEN MPI IS NOT USED, OR TO BE AVOIDED IN MPI RUNS\n"
    )
    mcgpu_infile.write(
        f"{threads_per_block} # GPU THREADS PER CUDA BLOCK (multiple of 32)\n"
    )
    mcgpu_infile.write(f"{histories_per_thread} # SIMULATED HISTORIES PER GPU THREAD\n")

    mcgpu_infile.write(f"\n\n")
    mcgpu_infile.write(f"#[SECTION SOURCE v.2011-07-12]\n")

    spctrm_mcgpu = None
    if spectrum == "60KV_AL35":
        spctrm_mcgpu = "60kVp_3.5mmAl.spc"
    elif spectrum == "90KV_AL40":
        spctrm_mcgpu = "90kVp_4.0mmAl.spc"
    elif spectrum == "120KV_AL43":
        spctrm_mcgpu = "120kVp_4.3mmAl.spc"
    else:
        log.exception("INVALID SPECTRUM NAME")
        return

    mcgpu_infile.write(f"../{spctrm_mcgpu} # X-RAY ENERGY SPECTRUM FILE\n")
    mcgpu_infile.write(
        f"{source_xyz_cm[0]} {source_xyz_cm[1]} {source_xyz_cm[2]} # SOURCE POSITION: X Y Z [cm]\n"
    )
    mcgpu_infile.write(
        f"{source_direction[0]} {source_direction[1]} {source_direction[2]} # SOURCE DIRECTION COSINES: U V W\n"
    )
    mcgpu_infile.write(
        f"-28.0 -58.0 # POLAR AND AZIMUTHAL APERTURES FOR THE FAN BEAM [degrees] (input negative to automatically cover the whole detector)\n"
    )

    mcgpu_infile.write(f"\n")
    mcgpu_infile.write(f"#[SECTION IMAGE DETECTOR v.2009-12-02]\n")
    mcgpu_infile.write(f"mcgpu_image_{filename}.dat # OUTPUT IMAGE FILE NAME\n")
    mcgpu_infile.write(
        f"{detector_pixels[0]} {detector_pixels[1]} # NUMBER OF PIXELS IN THE IMAGE: Nx Nz\n"
    )
    mcgpu_infile.write(
        f"{detector_size_cm[0]} {detector_size_cm[1]} # IMAGE SIZE (width, height): Dx Dz [cm]\n"
    )
    mcgpu_infile.write(
        f"{source_to_detector_distance_cm} # SOURCE-TO-DETECTOR DISTANCE (detector set in front of the source, perpendicular to the initial direction)\n"
    )

    mcgpu_infile.write(f"\n")
    mcgpu_infile.write(f"#[SECTION CT SCAN TRAJECTORY v.2011-10-25]\n")
    mcgpu_infile.write(
        f"1 # NUMBER OF PROJECTIONS (beam must be perpendicular to Z axis, set to 1 for a single projection)\n"
    )
    mcgpu_infile.write(
        f"45.0 # ANGLE BETWEEN PROJECTIONS [degrees] (360/num_projections for full CT)\n"
    )
    mcgpu_infile.write(
        f"-3590.99 3590.99 # ANGLES OF INTEREST (projections outside the input interval will be skipped)\n"
    )
    mcgpu_infile.write(
        f"60.0 # SOURCE-TO-ROTATION AXIS DISTANCE (rotation radius, axis parallel to Z)\n"
    )
    mcgpu_infile.write(
        f"0.0 # # VERTICAL TRANSLATION BETWEEN PROJECTIONS (HELICAL SCAN)\n"
    )

    mcgpu_infile.write(f"\n")
    mcgpu_infile.write(f"#[SECTION DOSE DEPOSITION v.2012-12-12]\n")
    mcgpu_infile.write(
        f"NO # TALLY MATERIAL DOSE? [YES/NO] (electrons not transported, x-ray energy locally deposited at interaction)\n"
    )
    mcgpu_infile.write(
        f"NO # TALLY 3D VOXEL DOSE? [YES/NO] (dose measured separately for each voxel)\n"
    )
    mcgpu_infile.write(f"mc-gpu_dose.dat # OUTPUT VOXEL DOSE FILE NAME\n")
    mcgpu_infile.write(
        f"1 {geom.shape[0]} # VOXEL DOSE ROI: X-index min max (first voxel has index 1)\n"
    )
    mcgpu_infile.write(f"1 {geom.shape[1]} # VOXEL DOSE ROI: Y-index min max\n")
    mcgpu_infile.write(f"1 {geom.shape[2]} # VOXEL DOSE ROI: Z-index min max\n")

    mcgpu_infile.write(f"\n")
    mcgpu_infile.write(f"#[SECTION VOXELIZED GEOMETRY FILE v.2009-11-30]\n")
    mcgpu_infile.write(
        f"{target_dir.as_posix()}/{filename}.vox # VOXEL GEOMETRY FILE (penEasy 2008 format; .gz accepted)\n"
    )

    mcgpu_infile.write(f"\n")
    mcgpu_infile.write(f"#[SECTION MATERIAL FILE LIST v.2009-11-30]\n")
    for mat_id in range(1, idx):
        mat_filename = get_mat_filename(id_mapping[mat_id])
        mcgpu_infile.write(
            f"../MC-GPU_material_files/{mat_filename}__5-120keV.mcgpu.gz # {mat_id}-th MATERIAL FILE (.gz accepted)\n"
        )

    mcgpu_infile.write("\n")
    mcgpu_infile.close()

