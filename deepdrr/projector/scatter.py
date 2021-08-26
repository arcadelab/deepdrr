#
# TODO: cite the papers that form the basis of this code
#

from typing import Tuple, Optional, Dict, List

import logging
import numpy as np
from . import spectral_data
from deepdrr import geo
from deepdrr import vol
from .rita import RITA
from .plane_surface import PlaneSurface

from .mcgpu_mfp_data import MFP_DATA
from .mcgpu_compton_data import MAX_NSHELLS as COMPTON_MAX_NSHELLS
from .mcgpu_compton_data import MATERIAL_NSHELLS, COMPTON_DATA

from .mcgpu_rita_samplers import rita_samplers

import math  # for 'count_milestones'
import time  # for keeping track of how long things take


log = logging.getLogger(__name__)


#
# USEFUL CONSTANTS
#

ELECTRON_REST_ENERGY = 510998.918  # eV

VOXEL_EPSILON = 0.000015
NEG_VOXEL_EPSILON = -0.000015


def make_woodcock_mfp(materials: List[str]) -> np.ndarray:
    """Generates and returns a table of [energy, Woodcock MFP] for each energy level, based on the provided materials

    Args:
        materials (List[str]): list of material names to generate Woodock MFP data for.  For a list of available materials, reference mcgpu_mfp_data.py
    
    Returns:
        np.ndarray: a table of [energy, Woodcock MFP] for each applicable energy level
    """
    # See http://serpent.vtt.fi/mediawiki/index.php/Delta-_and_surface-tracking, as well
    # as Woodcock et al. (1965) "Techniques Used in the GEM Code...", section 9.2
    # We assume that the volume is homogenous with the largest cross section of all the materials.
    # Accordingly, we assume that the mean free path is homogeneous with the shortest mean free
    # path of all materials

    mfp_woodcock = np.stack(
        (
            MFP_DATA["bone"][:, 0],  # the energy tables
            np.minimum.reduce([MFP_DATA[mat][:, 4] for mat in materials]),
        ),
        axis=1,
    )
    return mfp_woodcock


def simulate_scatter_no_vr(
    volume: vol.Volume,
    source_ijk: geo.Point3D,
    rt_kinv: np.ndarray,
    camera_intrinsics: geo.CameraIntrinsicTransform,
    source_to_detector_distance: float,
    index_from_ijk: np.ndarray,
    sensor_size: Tuple[int, int],
    photon_count: int,
    mfp_woodcock: np.ndarray,
    spectrum: Optional[np.ndarray] = spectral_data.spectrums["90KV_AL40"],
    E_abs: Optional[np.float32] = 5000,
) -> np.ndarray:
    """Produce a grayscale (intensity-based) image representing the photon scatter during an X-Ray, 
    without using VR (variance reduction) techniques.

    Args:
        volume (np.ndarray): the volume density data.
        source_ijk (geo.Point3D): the source point for rays in the camera's IJK space
        rt_kinv (np.ndarray): the ray transform for the projection.  Transforms pixel indices (u,v,1) to IJK vector along ray from from the X-Ray source to the detector pixel [u,v].
        camera_intrinsics (geo.CameraIntrinsicTransform): the C-Arm "camera" intrinsic transform.  Used to calculate the detector plane.
        source_to_detector_distance (float): distance from source to detector in millimeters.
        index_from_ijk (np.ndarray): the inverse transformation of ijk_from_index.  Takes 3D IJK coordinates and transforms to 2D pixel coordinates
        sensor_size (Tuple[int,int]): the sensor size {width}x{height}, in pixels, of the detector
        photon_count (int): the number of photons simulated.
        mfp_woodcock (np.ndarray): the Woodcock MFP data for the materials being simulated.  See make_woodock_mfp(...).
        spectrum (Optional[np.ndarray], optional): spectrum array.  Defaults to 90KV_AL40 spectrum.
        E_abs (Optional[np.float32], optional): the energy (in eV) at or below which photons are assumed to be absorbed by the materials.  Defaults to 5000 (eV).

    Returns:
        np.ndarray: deposited-energy image of the photon scatter
    """
    count_milestones = [
        int(math.pow(10, i))
        for i in range(int(1 + math.ceil(math.log10(photon_count))))
    ]  # [1, 10, 100, ..., 10^7] in default case

    accumulator = np.zeros(sensor_size).astype(np.float32)
    # return accumulator ### for when I don't really want to do noise stuff

    material_ids = {}
    for i, mat_name in enumerate(volume.materials.keys()):
        material_ids[i] = mat_name
    assert len(material_ids) > 0

    # Convert the volume segmentation data from one-hot to [0..N-1]-labeled
    assert np.all(np.equal(volume.data.shape, volume.materials[material_ids[0]].shape))
    labeled_seg = np.empty_like(volume.data)

    for ilabel in material_ids.keys():
        labeled_seg = np.add(
            labeled_seg, ilabel * volume.materials[material_ids[ilabel]]
        )

    # Plane data
    detector_plane = get_detector_plane(
        rt_kinv, camera_intrinsics, source_to_detector_distance, source_ijk, sensor_size
    )
    log.info(f"detector plane: {detector_plane}")
    ###
    log.debug(f"volume shape: {volume.data.shape}")
    log.debug(f"source_ijk: {source_ijk}")
    log.debug(f"Start time: {time.asctime()}")
    ###

    volume_min_bounds = (-0.5, -0.5, 0.5)
    volume_max_bounds = (
        volume.data.shape[0] - 0.5,
        volume.data.shape[1] - 0.5,
        volume.data.shape[2] - 0.5,
    )

    detector_hits = 0  # counts the number of photons that hit the detector
    volume_hits = 0
    pixel_hit_data = []
    for i in range(photon_count):
        if (i + 1) in count_milestones:
            log.debug(f"Simulating photon history {i+1} / {photon_count}")
            log.debug(f"\tCurrent time: {time.asctime()}")
        initial_dir = sample_initial_direction()
        hits_volume, initial_pos = move_photon_to_volume(
            source_ijk, initial_dir, volume_min_bounds, volume_max_bounds
        )
        if not hits_volume:
            continue
        volume_hits += 1
        initial_E = sample_initial_energy(spectrum)
        pixel_x, pixel_y, energy, num_scatter_events = track_single_photon_no_vr(
            initial_pos,
            initial_dir,
            initial_E,
            E_abs,
            labeled_seg,
            volume.data.shape,
            detector_plane,
            index_from_ijk,
            source_ijk,
            source_to_detector_distance,
            mfp_woodcock,
            material_ids,
        )

        # Only keep track of photons that were scattered
        if 0 == num_scatter_events:
            continue
        # else:
        # log.debug(f"photon history {i+1} / {photon_count}: {num_scatter_events} scatter events")
        # log.debug(f"\tpixel: [{pixel_x}, {pixel_y}]\n")

        # Model for detector: ideal image formation
        # Each pixel counts the total energy of the X-rays that enter the pixel (100% efficient pixels)
        # NOTE: will have to take care that this sort of tallying is <<<ATOMIC>>> in the GPU
        if (pixel_x < 0) or (pixel_x >= sensor_size[0]):
            continue
        if (pixel_y < 0) or (pixel_y >= sensor_size[1]):
            continue
        pixel_hit_data.append((pixel_x, pixel_y, initial_E, energy))
        accumulator[pixel_x, pixel_y] = accumulator[pixel_x, pixel_y] + energy
        detector_hits += 1

    log.info(
        f"Finished simulating {photon_count} photon histories.  {detector_hits} / {photon_count} photons hit the detector"
    )
    log.info("pixel hit data: [pixel_x, pixel_y], initial_energy -> energy")
    for tup in pixel_hit_data:
        log.info(f"\t[{tup[0]}, {tup[1]}], {tup[2]} -> {tup[3]}")
    ###return accumulator
    return accumulator, pixel_hit_data


def track_single_photon_no_vr(
    initial_pos: geo.Point3D,
    initial_dir: geo.Vector3D,
    initial_E: np.float32,
    E_abs: np.float32,
    labeled_seg: np.ndarray,
    volume_shape: Tuple[int, int, int],
    detector_plane: PlaneSurface,
    index_from_ijk: np.ndarray,
    source_ijk: geo.Point3D,
    source_to_detector_distance: float,
    mfp_woodcock: np.ndarray,
    material_ids: Dict[int, str],
) -> Tuple[int, int, np.float32, int]:
    """Produce a grayscale (intensity-based) image representing the photon scatter of a single photon
    during an X-Ray, without using VR (variance reduction) techniques.

    Args:
        initial_pos (geo.Point3D): the initial position (in IJK space) of the photon once it has entered the volume.  This IS NOT the X-Ray source.  See function sample_initial_direction(...)
        initial_dir (geo.Vector3D): the initial direction of travel of the photon, in IJK space
        initital_E (np.float32): the initial energy of the photon
        E_abs (np.float32): the energy (in eV) at or below which photons are assumed to be absorbed by the materials.
        labeled_seg (np.ndarray): a [0..N-1]-labeled segmentation of the volume
        density_vol (np.ndarray): the density information of the volume
        detector_plane (np.ndarray): the 'plane vector' of the detector
        index_from_ijk (np.ndarray): the inverse transformation of ijk_from_index, the ray transform for the projection.
        material_ids (Dict[int,str]): a dictionary mapping an integer material ID-label to the name of the material
    Returns:
        Tuple[int, int, np.float32, int]: the pixel coord.s of the hit pixel, as well as the energy (in eV) of the photon when it hit the detector.  
                                    The final int is the number of scatter events experienced by the photon.
                                    Note that the returned pixel coord.s CAN BE out-of-bounds.
    """
    pos = initial_pos
    direction = initial_dir

    # log.info(f"initial_pos: {initial_pos}")
    # log.info(f"initial_dir: {initial_dir}")

    photon_energy = (
        initial_E  # tracker variable throughout the duration of the photon history
    )

    num_scatter_events = 0

    vox_x, vox_y, vox_z = None, None, None

    while True:  # emulate a do-while loop
        # Get voxel (index) coord.s.  Keep in mind that IJK coord.s are voxel-centered
        vox_x = int(
            np.floor(pos.data[0] + 0.5)
        )  # shift because volume's IJK bounds are [-0.5, {x,y,z}_len - 0.5]
        vox_y = int(np.floor(pos.data[1] + 0.5))
        vox_z = int(np.floor(pos.data[2] + 0.5))
        # log.debug(f"voxel: ({vox_x}, {vox_y}, {vox_z})")
        if (vox_x < 0) or (vox_x >= volume_shape[0]):
            break
        if (vox_y < 0) or (vox_y >= volume_shape[1]):
            break
        if (vox_z < 0) or (vox_z >= volume_shape[2]):
            break
        # voxel_coords = (vox_x, vox_y, vox_z)
        # log.debug(f"INSIDE VOLUME")
        mat_label = labeled_seg[vox_x, vox_y, vox_z]
        mat_name = material_ids[mat_label]

        mfp_wc = get_woodcock_mfp(mfp_woodcock, photon_energy)
        # mfp_Ra, mfp_Co, mfp_Tot = get_mfp_data(mfp_data[mat_name], photon_energy)

        # Delta interactions
        while True:
            # simulate moving the photon
            s = (
                -1 * (10 * mfp_wc) * np.log(sample_U01())
            )  # multiply by 10 to convert from MFP data (cm) to voxel spacing (mm)
            pos = geo.Point3D.from_any(pos + (s * direction))

            # Check for leaving the volume
            vox_x = int(
                np.floor(pos.data[0] + 0.5)
            )  # shift because volume's IJK bounds are [-0.5, {x,y,z}_len - 0.5]
            vox_y = int(np.floor(pos.data[1] + 0.5))
            vox_z = int(np.floor(pos.data[2] + 0.5))
            # log.debug(f"voxel: ({vox_x}, {vox_y}, {vox_z})")
            if (vox_x < 0) or (vox_x >= volume_shape[0]):
                break
            if (vox_y < 0) or (vox_y >= volume_shape[1]):
                break
            if (vox_z < 0) or (vox_z >= volume_shape[2]):
                break

            mat_label = labeled_seg[vox_x, vox_y, vox_z]
            mat_name = material_ids[mat_label]

            mfp_Ra, mfp_Co, mfp_Tot = get_mfp_data(MFP_DATA[mat_name], photon_energy)

            # log.debug(f"probability to accept the collision: mfp_wc / mfp_Tot == {mfp_wc / mfp_Tot}")

            if sample_U01() < mfp_wc / mfp_Tot:
                # Accept the collision.  See http://serpent.vtt.fi/mediawiki/index.php/Delta-_and_surface-tracking
                break
            # log.debug(f"DELTA COLLISION")

        # might have left the volume OR had a legitimate interaction
        if (vox_x < 0) or (vox_x >= volume_shape[0]):
            break
        if (vox_y < 0) or (vox_y >= volume_shape[1]):
            break
        if (vox_z < 0) or (vox_z >= volume_shape[2]):
            break

        # Now at a legitimate photon interaction

        # Sample the photon interaction type
        #
        # (1 / mfp_Tot) * (1 / molecules_per_vol) ==    total interaction cross section =: sigma_Tot
        # (1 / mfp_Ra ) * (1 / molecules_per_vol) == Rayleigh interaction cross section =: sigma_Ra
        # (1 / mfp_Co ) * (1 / molecules_per_vol) ==  Compton interaction cross section =: sigma_Co
        #
        # SAMPLING RULE: Let rnd be a uniformly selected number on [0,1]
        #
        # if rnd < (simga_Co / sigma_Tot): # if rnd < (mfp_Tot / mfp_Co)
        #   COMPTON INTERACTION
        # elif rnd < (sigma_Ra + sigma_Co) / sigma_Tot: # if rnd < mfp_Tot * ((1 / mfp_Co) + (1 / mfp_Ra))
        #   RAYLEIGH INTERACTION
        # else:
        #   OTHER INTERACTION (photoelectric for pair production) ==> photon absorbed
        #
        cos_theta, E_prime = None, None
        rnd = sample_U01()
        if rnd < (mfp_Tot / mfp_Co):
            cos_theta, E_prime = sample_Compton_theta_E_prime(
                photon_energy, MATERIAL_NSHELLS[mat_name], COMPTON_DATA[mat_name]
            )
        elif rnd < mfp_Tot * ((1 / mfp_Co) + (1 / mfp_Ra)):
            cos_theta = sample_Rayleigh_theta(photon_energy, rita_samplers[mat_name])
            E_prime = photon_energy
        else:
            # Photoelectric interaction OR pair production.  Photon is absorbed, and thus does not hit the detector.
            return -1, -1, photon_energy, num_scatter_events

        photon_energy = E_prime
        if photon_energy <= E_abs:
            return -1, -1, photon_energy, num_scatter_events

        num_scatter_events += 1
        # log.debug(f"SCATTER EVENT")

        phi = 2 * np.pi * sample_U01()
        direction = get_scattered_dir(direction, cos_theta, phi)

        # END WHILE

    # final processing

    # log.debug(f"pos after leaving volume: {pos}")
    # log.debug(f"dir after leaving volume: {direction}")

    # Transport the photon to the detector plane
    hits_detector_dist = detector_plane.check_ray_intersection(pos, direction)
    # log.debug(f"hits_detector_dist: {hits_detector_dist}")
    if (hits_detector_dist is None) or (hits_detector_dist < 0.0):
        # log.debug("NO HIT")
        return -1, -1, photon_energy, num_scatter_events

    hit = geo.Point3D.from_any(pos + (hits_detector_dist * direction))

    # log.debug(f"hit: {hit}")

    # NOTE: an alternative formulation would be to use (rt_kinv).inv == index_from_ijk
    pixel_x, pixel_y = detector_plane.get_lin_comb_coefs(hit)
    # log.debug(f"old pixel: {pixel_x}, {pixel_y}")

    hit_x = (hit.data[0] - source_ijk.data[0]) / source_to_detector_distance
    hit_y = (hit.data[1] - source_ijk.data[1]) / source_to_detector_distance
    hit_z = (hit.data[2] - source_ijk.data[2]) / source_to_detector_distance
    pixel_x = (
        index_from_ijk[0, 0] * hit_x
        + index_from_ijk[0, 1] * hit_y
        + index_from_ijk[0, 2] * hit_z
    )
    pixel_y = (
        index_from_ijk[1, 0] * hit_x
        + index_from_ijk[1, 1] * hit_y
        + index_from_ijk[1, 2] * hit_z
    )

    # log.debug(f"new pixel: {pixel_x}, {pixel_y}")

    return (
        int(np.floor(pixel_x)),
        int(np.floor(pixel_y)),
        photon_energy,
        num_scatter_events,
    )


def get_mfp_data(
    table: np.ndarray, E: np.float32
) -> Tuple[np.float32, np.float32, np.float32]:
    """Access the Mean Free Path data for the given material's table at the given photon energy level.
    Performs linear interpolation for any energy value that isn't exactly a table entry.

    Args:
        table (np.ndarray): a table of Mean Free Path data.  See mcgpu_mean_free_path_data directory for examples.
        E (np.float32): the energy of the photon
    
    Returns:
        np.float32: the Rayleigh scatter mean free path
        np.float32: the Compton scatter mean free path
        np.float32: the total mean free path
    """
    # Binary search to find the proper table entry.  Want energy(lo_bin) <= E < energy(hi_bin), with (lo_bin + 1) == hi_bin
    lo_idx = 0  # inclusive
    hi_idx = table.shape[0]  # exclusive
    i = None  # the index of the bin that we find E in

    while lo_idx < hi_idx:
        mid_idx = np.floor_divide(lo_idx + hi_idx, np.int32(2))

        if E < table[mid_idx, 0]:
            # Need to check lower intervals
            hi_idx = mid_idx
        elif E < table[mid_idx + 1, 0]:
            # found correct interval
            i = mid_idx
            break
        else:
            # Need to check higher intervals
            lo_idx = mid_idx + 1

    assert (table[i, 0] <= E) and (E < table[i + 1, 0])

    # Linear interpolation for each of the three values
    delta_E = table[i + 1, 0] - table[i, 0]
    partial = E - table[i, 0]

    delta_mfp_Ra = table[i + 1, 1] - table[i, 1]
    delta_mfp_Co = table[i + 1, 2] - table[i, 2]
    delta_mfp_Tot = table[i + 1, 4] - table[i, 4]

    mfp_Ra = table[i, 1] + (delta_mfp_Ra * partial) / delta_E
    mfp_Co = table[i, 1] + (delta_mfp_Co * partial) / delta_E
    mfp_Tot = table[i, 1] + (delta_mfp_Tot * partial) / delta_E

    return mfp_Ra, mfp_Co, mfp_Tot


def get_woodcock_mfp(table: np.ndarray, E: np.float32) -> np.float32:
    """Access the Woodcock Mean Free Path at the given photon energy level.
    For an explanation of what the Woodcock Mean Free Path is, see mcgpu_mfp_data.py.
    Performs linear interpolation for any energy value that isn't exactly a table entry.

    Args:
        table (np.ndarray): a table of Woodcock Mean Free Path data.  See make_woodcock_mfp(...).
        E (np.float32): the energy of the photon
    
    Returns:
        np.float32: the inverse of the total majorant cross section.  This returned value has units of centimeters.
    """
    # Binary search to find the proper table entry.  Want energy(lo_bin) <= E < energy(hi_bin), with (lo_bin + 1) == hi_bin
    lo_idx = 0  # inclusive
    hi_idx = table.shape[0]  # exclusive
    i = None  # the index of the bin that we find E in

    while lo_idx < hi_idx:
        mid_idx = np.floor_divide(lo_idx + hi_idx, np.int32(2))

        if E < table[mid_idx, 0]:
            # Need to check lower intervals
            hi_idx = mid_idx
        elif E < table[mid_idx + 1, 0]:
            # found correct interval
            i = mid_idx
            break
        else:
            # Need to check higher intervals
            lo_idx = mid_idx + 1

    assert (table[i, 0] <= E) and (E < table[i + 1, 0])

    # Linear interpolation
    delta_E = table[i + 1, 0] - table[i, 0]
    partial = E - table[i, 0]

    delta_mfp_Tot = table[i + 1, 1] - table[i, 1]

    mfp_wc = table[i, 1] + (delta_mfp_Tot * partial) / delta_E

    return mfp_wc


def sample_initial_direction() -> geo.Vector3D:
    """Returns an initial direction vector for a photon, uniformly distributed over the unit sphere.

    Returns:
        geo.Vector3D: the initial direction unit vector (dx, dy, dz)^T
    """
    phi = 2 * np.pi * sample_U01()  # azimuthal angle
    theta = np.arccos(1 - 2 * sample_U01())  # polar angle

    sin_theta = np.sin(theta)

    dx = sin_theta * np.cos(phi)
    dy = sin_theta * np.sin(phi)
    dz = np.cos(theta)

    direction = geo.Vector3D.from_array(np.array([dx, dy, dz]))
    return direction


def move_photon_to_volume(
    pos: geo.Point3D,
    direction: geo.Vector3D,
    volume_min_bounds: Tuple[float, float, float],
    volume_max_bounds: Tuple[float, float, float],
) -> Tuple[bool, geo.Point3D]:
    """Transports a photon at the given position, travelling in the given direction, to a rectangular-prism volume of the given bounds.
    Assumes the volume's surfaces are aligned with the major planes of the coordinate system

    Args:
        pos (geo.Point3D): the initial position of the photon.  Very likely to be the X-ray source.
        direction (geo.Vector3D): a unit vector denoting the direction in which the photon is traveling
        volume_min_bounds (Tuple[float, float, float]): the minimum coordinate bound for the volume in each direction
        volume_max_bounds (Tuple[float, float, float]): the minimum coordinate bound for the volume in each direction
    
    Returns:
        bool: whether the photon hits the volume or not
        geo.Point3D: where the photon hits the volume if it hits the volume, else the original position
    """
    pos_x = pos.data[0]
    pos_y = pos.data[1]
    pos_z = pos.data[2]

    dir_x = direction.data[0]
    dir_y = direction.data[1]
    dir_z = direction.data[2]

    min_x = volume_min_bounds[0]
    min_y = volume_min_bounds[1]
    min_z = volume_min_bounds[2]

    max_x = volume_max_bounds[0]
    max_y = volume_max_bounds[1]
    max_z = volume_max_bounds[2]

    dist_x, dist_y, dist_z = None, None, None

    # x-direction calculations
    if dir_x > VOXEL_EPSILON:
        if pos_x > min_x:  # photon inside or past volume
            dist_x = 0.0
        else:  # add VOXEL_EPSILON to make super sure that the photon reaches the volume
            dist_x = VOXEL_EPSILON + (min_x - pos_x) / dir_x
    elif dir_x < NEG_VOXEL_EPSILON:
        if pos_x < max_x:
            dist_x = 0.0
        else:
            dist_x = VOXEL_EPSILON + (max_x - pos_x) / dir_x
    else:
        dist_x = float("-inf")

    # y-direction calculations
    if dir_y > VOXEL_EPSILON:
        if pos_y > min_y:  # photon inside or past volume
            dist_y = 0.0
        else:  # add VOXEL_EPSILON to make super sure that the photon reaches the volume
            dist_y = VOXEL_EPSILON + (min_y - pos_y) / dir_y
    elif dir_x < NEG_VOXEL_EPSILON:
        if pos_x < max_y:
            dist_y = 0.0
        else:
            dist_y = VOXEL_EPSILON + (max_y - pos_y) / dir_y
    else:
        dist_y = float("-inf")

    # z-direction calculations
    if dir_z > VOXEL_EPSILON:
        if pos_z > min_z:  # photon inside or past volume
            dist_z = 0.0
        else:  # add VOXEL_EPSILON to make super sure that the photon reaches the volume
            dist_z = VOXEL_EPSILON + (min_z - pos_z) / dir_z
    elif dir_z < NEG_VOXEL_EPSILON:
        if pos_z < max_z:
            dist_z = 0.0
        else:
            dist_z = VOXEL_EPSILON + (max_z - pos_z) / dir_z
    else:
        dist_z = float("-inf")

    max_dist = max([dist_x, dist_y, dist_z])

    new_pos_x = pos_x + (max_dist * dir_x)
    new_pos_y = pos_y + (max_dist * dir_y)
    new_pos_z = pos_z + (max_dist * dir_z)

    if (
        (new_pos_x < min_x)
        or (new_pos_x > max_x)
        or (new_pos_y < min_y)
        or (new_pos_y > max_y)
        or (new_pos_z < min_z)
        or (new_pos_z > max_z)
    ):
        return False, geo.Point3D.from_array(np.array([pos_x, pos_y, pos_z]))
    else:
        return True, geo.Point3D.from_array(np.array([new_pos_x, new_pos_y, new_pos_z]))


def get_detector_plane(
    rt_kinv: np.ndarray,
    camera_intrinsics: geo.CameraIntrinsicTransform,
    sdd: float,
    source_world: geo.Point3D,
    sensor_size: Tuple[int, int],
) -> PlaneSurface:
    """Calculates the PlaneSurface object of the detector plane in IJK coordinates.
    Note that the cosines of the plane's normal vector (n_x, n_y, n_z) are NOT normalized to be a unit vector.

    The first basis vector represents moving one pixel ACROSS the image (left to right).
    The second basis vector represents moving one pixel DOWN the image (top to bottom).

    Args:
        rt_kinv (np.ndarray): the 3x3 ray transform for the projection.  Transforms pixel indices (u,v,1) to world-space vector along
                            the ray from the X-Ray source to the pixel [u,v] on the detector, such that the resulting world-space vector
                            has unit projection along the vector pointing from the source to the center of the detector.
        camera_intrinsics (geo.CameraIntrinsicTransform): the 3x3 matrix that denotes the camera's intrinsics.  Canonically represented by K.
        sdd (float): the distance from the X-Ray source to the detector.
        source_world (geo.Point3D): the world coordinates of the X-Ray source
        sensor_size (Tuple[int,int]): the sensor size {width}x{height}, in pixels, of the detector

    Returns:
        PlaneSurface: a PlaneSurface object representing the detector.  
    """
    # Based off the project_kernel.cu code:
    #   Let \hat{p} = (u,v,1)^T be the pixel coord.s on the detector plane
    #   Then, the 3D world coord.s of that pixel are related to (R^T K^{-1}) \hat{p} == (rt_kinv) @ \hat{p}
    #   Specifically, (rt_kinv) @ \hat{p} is a world vector along the ray from the X-Ray source to the
    #   pixel (u,v) on the detector plane.  Since, after investigation, I found that the vector
    #   [(rt_kinv) @ (W/2,H/2,1)^T] always has magnitude 1.00000, the vector:
    #       SDD * (rt_kinv) @ (u,v,1)^T
    #   points from the X-Ray source to the pixel (u,v) on the detector plane, where SDD is the
    #   source-to-detector distance.
    #
    # We calculate the normal vector of the detector plane in world-space by using the three-point method:
    #   1. Let {p1, p2, p3} be three pixel coordinates of the form (u, v, 1)^T
    #   2. Three coplanar points in world coordinates are r1 := SDD * (rt_kinv) @ p1, r2 := SDD * (rt_kinv) @ p2,
    #      r3 := SDD * (rt_kinv) @ p3
    #   3. Compute two vectors that are -in- the plane: v1 := r2 - r1, v2 := r3 - r1
    #   4. The cross product v := v1 x v2 is perpendicular to both v1 and v2.  Thus, v is a normal vector to the plane
    #
    # Note that even though {r1, r2, r3} are technically the vectors points from the X-ray source to the detector plane,
    # not pointing from the world origin to the detector plane, the fact that {v1, v2} are [relative displacement vectors]
    # means that the shift in "origin" for {r1, r2, r3} has no effect on calculating the normal vector for the detector plane.
    #
    # Simplifying the math to reduce the number of arithmetic steps:
    #   v1 = r2 - r1 = SDD * (rt_kinv) @ p2 - SDD * (rt_kinv) @ p1 = SDD * (rt_kinv) @ (p2 - p1)
    #   v2 = r3 - r1 = SDD * (rt_kinv) @ p3 - SDD * (rt_kinv) @ p1 = SDD * (rt_kinv) @ (p3 - p1)
    #
    # Choosing easy p_i's of: p1 = (0, 0, 1)^T, p2 = (1, 0, 1)^T, p3 = (0, 1, 1)^T, we get:
    #   v1 = SDD * (rt_kinv) @ (1, 0, 0)^T = SDD * [first column of rt_kinv]  // corresponds to moving 1 pixel over in x-direction (x increases)
    #   v2 = SDD * (rt_kinv) @ (0, 1, 0)^T = SDD * [second column of rt_kinv] // corresponds to moving 1 pixel down in y-direction (y increases)
    #
    # To reduce the number of characters, let M: = (rt_kinv), as a 9-element row-major ordering of the 3x3 (rt_kinv).
    #   v := v1 x v2 = [SDD * (M[0], M[3], M[6])^T] x [SDD * (M[1], M[4], M[7])^T]
    #       = (SDD * SDD) * [(M[0], M[3], M[6])^T x (M[1], M[4], M[7])^T]
    #       = SDD * SDD * (
    #           M[3] * M[7] - M[6] * M[4],
    #           M[6] * M[1] - M[0] * M[7],
    #           M[0] * M[4] - M[3] * M[1]
    #         )^T
    #       = SDD * SDD* (
    #           rt_kinv[1,0] * rt_kinv[2,1] - rt_kinv[2,0] * rt_kinv[1,1],
    #           rt_kinv[2,0] * rt_kinv[0,1] - rt_kinv[0,0] * rt_kinv[2,1],
    #           rt_kinv[0,0] * rt_kinv[1,1] - rt_kinv[1,0] * rt_kinv[0,1]
    #         )^T
    #
    # Once we have the normal vector, we need the minimum distance between the detector plane and
    # the origin of the world coord.s to get the fourth entry in the 'plane vector' (n_x, n_y, n_z, d)
    #
    sdd_sq = sdd * sdd

    # Normal vector to the detector plane:
    nx = sdd_sq * (rt_kinv[1, 0] * rt_kinv[2, 1] - rt_kinv[2, 0] * rt_kinv[1, 1])
    ny = sdd_sq * (rt_kinv[2, 0] * rt_kinv[0, 1] - rt_kinv[0, 0] * rt_kinv[2, 1])
    nz = sdd_sq * (rt_kinv[0, 0] * rt_kinv[1, 1] - rt_kinv[1, 0] * rt_kinv[0, 1])

    n_mag = np.sqrt((nx * nx) + (ny * ny) + (nz * nz))

    nx /= n_mag
    ny /= n_mag
    nz /= n_mag

    # The 'surface origin' corresponds to the pixel [0,0] on the detector.
    # Vector source_to_surf_ori = SDD * (rt_kinv) @ (0,0,1)^T = SDD * [third column of rt_kinv]
    surf_ori_x = (sdd * rt_kinv[0, 2]) + source_world.data[
        0
    ]  # source_to_surf_ori + origin_to_source == origin_to_surf_ori
    surf_ori_y = (sdd * rt_kinv[1, 2]) + source_world.data[1]
    surf_ori_z = (sdd * rt_kinv[2, 2]) + source_world.data[2]
    # SANITY CHECK: after using an inverse-of-upper-triangular-matrix formula, we get:
    # kinv[2] == (s c_y - c_x f_y) / (f_x f_y)
    # kinv[5] == c_y / f_y
    # kinv[8] == 1
    surface_origin = geo.Point3D.from_array(
        np.array([surf_ori_x, surf_ori_y, surf_ori_z])
    )

    # Distance from the detector plane to the origin
    #
    # Time for a diagram.
    #
    #       |   plane parallel to detector plane
    #       |             v
    #       |             |
    #     X |  -----_____ |
    #       |            -O---_____
    #       |             |        -----_____
    #     C | ------------P-----------------------  S
    #       |             |
    #       |             |
    #       |
    #       ^
    # detector plane
    #
    # Points:
    #   - S is the SOURCE of the X-rays
    #   - O is the ORIGIN of the volume
    #   - C is the CENTER of the detector
    #   - X is the point where the ray from S to O intersects the detector plane
    #   - P is a point along the ray from S to C such that triangle SPO is similar to triangle SCX
    #
    # Vector SC is always perpendicular to the detector plane.
    #
    # The shortest distance between the detector plane and the origin O is the magnitude of a vector
    # from O to the dectector plane, where that vector from O is perpendicular to the detector plane.
    # Thus,
    #
    #   d = magnitude(projection of OX onto SC)
    #     = magnitude(SC * (OX \cdot SC) / (magnitude(SC)^2))
    #     = magnitude(SC) * (OX \cdot SC) / (magnitude(SC)^2)
    #     = (OX \cdot SC) / magnitude(SC)
    #
    # Vector SC can be found by finding the world coordinates of the detector center (vector OC) and
    # the world coordinates of the X-ray source (vector OS).  SC = OC - OS
    #
    # Vector SP = projection of SO onto SC
    #           = SC * (SO \cdot SC) / (magnitude(SC)^2)
    # magnitude(SP) = (SO \cdot SC) / magnitude(SC)
    #
    # Vector SX = SO * magnitude(SC) / magnitude(SP) = SO * (SC \cdot SC) / (SO \cdot SC)
    # Vector OX = SX - SO
    #           = SO * ([(SC \cdot SC) / (SO \cdot SC)] - 1)
    #           = OS * (-1) * ([(SC \cdot SC) / ((-1) OS \cdot SC)] - 1)
    #           = OS * ([(SC \cdot SC) / (OS \cdot SC)] + 1)
    #
    # Once we know vector OX, we can plug it into the above formula for 'd'.
    #
    # d = (OX \cdot SC) / magnitude(SC)
    #   = ({OS * ([(SC \cdot SC) / (OS \cdot SC)] + 1)} \cdot SC) / magnitude(SC)
    #   = ([(SC \cdot SC) / (OS \cdot SC)] + 1) * (OS \cdot SC) / magnitude(SC)
    #   = magnitude(SC) + [(SC \cdot OS) / magitude(SC)]
    #   = magnitude(SC) * [1 + (SC \cdot OS) / (SC \cdot SC)]
    #
    cu = sensor_size[0] / 2  # pixel coord.s of the center of the detector
    cv = sensor_size[1] / 2

    # world coord.s of the SC, the ray from the X-Ray source S to the center C of the detector
    sc_x = sdd * (cu * rt_kinv[0, 0] + cv * rt_kinv[0, 1] + rt_kinv[0, 2])
    sc_y = sdd * (cu * rt_kinv[1, 0] + cv * rt_kinv[1, 1] + rt_kinv[1, 2])
    sc_z = sdd * (cu * rt_kinv[2, 0] + cv * rt_kinv[2, 1] + rt_kinv[2, 2])

    # Note that the world coord.s of vector OS are contained in source_world, which is obtained
    # by calling the camera_center_in_volume method
    sc_dot_sc = (sc_x * sc_x) + (sc_y * sc_y) + (sc_z * sc_z)
    sc_dot_os = (
        (sc_x * source_world.data[0])
        + (sc_y * source_world.data[1])
        + (sc_z * source_world.data[2])
    )

    # the distance to the detector from the origin -- the absolute value is important!
    d = abs(np.sqrt(sc_dot_sc) * (1 + (sc_dot_os / sc_dot_sc)))

    # The normal vector (nx,ny,nz)^T points either:
    #   1) from the detector plane TOWARD the origin
    #   2) from the detector plane AWAY from the origin
    #
    # If the normal vector points TOWARD the origin, then the plane equation is:
    #   [position] \cdot [normal vector] + [distance to detector] = 0,
    # which suggests that the fourth component of the 'plane vector' should be positive.
    #
    # If the normal vector points AWAY from the origin, then the plane equation is:
    #   [position] \cdot [normal vector] - [distance to detector] = 0,
    # which suggests that the fourth component of the 'plane vector' should be negative.
    #
    # Suppose we are viewing the principal axis of the C-Arm (the line that goes from the source
    # and hits the center of the detector at a perpendicular angle) such that the detector is on
    # the left and the source is on the right:
    #
    #      detector plane
    #          v                                 plane perpendicular to source
    #          |                                              v
    #          |                                              |
    #          |             principal axis                   |
    #          |                   v                          |
    #        C |--------------------------------------------- @ S (source)
    #          |                                              |
    #          |                                              |
    #          |                                              |
    #          |
    #
    # There are five cases. Let P be the point resulting from projecting the origin onto the
    # line from S to C -- {S,C,P} are three collinear points.  This point P is the same as from
    # the above explanation from determining the origin-to-plane distance.
    #   1. C between P and S.  Use the [-d] equation iff SC and \hat{n} are opposite directions.
    #   2. P coincides with C.  d == (-d) == 0, so nothing needs to happen.
    #   3. P between C and S.  Use the [-d] equation iff SC and \hat{n} are the same direction.
    #   4. P coincides with S.  Use the [-d] equation iff SC and \hat{n} are the same direction.
    #   5. S between C and P.  Use the [-d] equation iff SC and \hat{n} are the same direction.
    #
    # However, these cases can be re-parameterized by SC and SP:
    #   1. [(SC dot SP) > 0] and [mag(SP) > mag(SC)]
    #   2. [(SC dot SP) > 0] and [mag(SP) = mag(SC)]
    #   3. [(SC dot SP) > 0] and [mag(SP) < mag(SC)]
    #   4. [(SC dot SP) = 0]
    #   5. [(SC dot SP) < 0]
    #
    # However, since we have already calculated (SC dot OS) previously, we note that (SC dot SP)
    # is equivalent to [-1 * (SC dot OS)].  Addtionally, we note that magnitude comparisons are
    # the same if we compare the magnitude-squared, and:
    #   [magnitude(SP)^2] == [magnitude(-1 * SC * (SC dot OS) / (SC dot SC))^2]
    #                     == (SC dot OS)^2 / (SC dot SC)
    #
    # Re-parameterizing again, we get:
    #   1. [(SC dot OS) < 0] and [(SC dot OS)^2 / (SC dot SC) > (SC dot SC)]
    #   2. [(SC dot OS) < 0] and [(SC dot OS)^2 / (SC dot SC) = (SC dot SC)]
    #   3. [(SC dot OS) < 0] and [(SC dot OS)^2 / (SC dot SC) < (SC dot SC)]
    #   4. [(SC dot OS) = 0]
    #   5. [(SC dot OS) > 0]
    # Simplifying:
    #   1. [(SC dot OS) < 0] and [ABS(SC dot OS) > (SC dot SC)]
    #   2. [(SC dot OS) < 0] and [ABS(SC dot OS) = (SC dot SC)]
    #   3. [(SC dot OS) < 0] and [ABS(SC dot OS) < (SC dot SC)]
    #   4. [(SC dot OS) = 0]
    #   5. [(SC dot OS) > 0]
    # where ABS(...) is the absolute value function.
    #
    # For the actual implementation, we recall that Case 1 has one behavior, while Cases 2-5 share
    # the other behavior.
    #
    sc_dot_normal = (sc_x * nx) + (sc_y * ny) + (sc_z * nz)

    if (sc_dot_os < 0) and (
        (-sc_dot_os) > sc_dot_sc
    ):  # Uses ABS(sc_dot_os) == -sc_dot_os
        # CASE 1.  Use [-d] equation iff SC and \hat{n} are opposite directions
        if sc_dot_normal < 0:
            d = -d  # d was previously necessarily positive semidefinite
    else:
        # CASES 2,3,4,5.  Use the [-d] equation iff SC and \hat{n} are the same direction.
        if sc_dot_normal > 0:
            d = -d  # d was previously necessarily positive semidefinite

    plane_vector = np.array([nx, ny, nz, d])

    # The basis is {v1, v2}, where {v1, v2} are the vectors described in the "choosing easy p_i's" section
    # That way, the point of intersection is:
    #
    #   intersection = surface_origin + (pixel_x_value) * v1 + (pixel_y_value) * v2
    #
    v1 = geo.Vector3D.from_array(sdd * rt_kinv[:, 0])
    v2 = geo.Vector3D.from_array(sdd * rt_kinv[:, 1])

    # Coordinate bounds correspond to the size of the detector, in pixels
    bounds = np.array([[0, sensor_size[0]], [0, sensor_size[1]]])

    # Determine whether or not the basis vectors are orthogonal
    #
    # v1 = SDD * (rt_kinv) @ (1,0,0)^T = SDD * [R^T] @ [K.inv] @ (1,0,0)^T
    # v2 = SDD * (rt_kinv) @ (0,1,0)^T = SDD * [R^T] @ [K.inv] @ (0,1,0)^T
    #
    #     [f_x  s  c_x]                     [1/f_x -s/(f_x f_y)  ...]
    # K = [ 0  f_y c_y]     ==>    K^{-1} = [  0       1/f_y     ...]
    #     [ 0   0   1 ]                     [  0         0        1 ]
    #
    # where the "..." in the third column are left incomplete because the
    # calculation does not involve the third column of K^{-1}.
    #
    # (v1)^T (v2) = [SDD * [R^T] @ [K.inv] @ (1,0,0)^T]^T [SDD * [R^T] @ [K.inv] @ (0,1,0)^T]
    #             = (SDD^2) * [K.inv @ (1,0,0)^T]^T @ R @ R^T @ [K.inv @ (0,1,0)^T]
    #             = (SDD^2) * [K.inv @ (1,0,0)^T]^T @ [K.inv @ (0,1,0)^T]
    #             = (SDD^2) * (1/f_x, 0, 0)^T @ (-s/(f_x f_y), 1/f_y, 0)
    #             = (SDD^2) * (-s / (f_x f_x f_y))
    #
    # Thus, {v1,v2} are orthogonal iff 's', the pixel shear, is zero.
    #
    shear = camera_intrinsics.data[0, 1]
    return PlaneSurface(plane_vector, surface_origin, (v1, v2), bounds, (0 == shear))


def get_scattered_dir(
    direction: geo.Vector3D, cos_theta: np.float32, phi: np.float32
) -> geo.Vector3D:
    """Determine the new direction of travel after getting scattered

    Args:
        dir (geo.Vector3D): the incoming direction of travel
        cos_theta (np.float32): the cosine of the polar scattering angle, i.e. the angle dir and dir_prime
        phi (np.float32): the azimuthal angle, i.e. how dir_prime is rotated about the axis 'dir'.

    Returns:
        geo.Vector3D: the outgoing direction of travel
    """
    dx = direction.data[0]
    dy = direction.data[1]
    dz = direction.data[2]

    # since \theta is restricted to [0, \pi], sin_theta is restricted to [0,1]
    sin_theta = np.sqrt(1 - cos_theta * cos_theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    tmp = np.sqrt(1 - dz * dz)
    if tmp < 0.000001:
        log.debug(f"In get_scattered_dir(...)")
        log.debug(f"\tinput direction: ({dx}, {dy}, {dz})")
        log.debug(f"1 - dz*dz = {tmp} < 0.000001")
        tmp = 0.0000001

    # See PENELOPE-2006 Eqn 1.131
    new_dx = dx * cos_theta + sin_theta * (dx * dz * cos_phi - dy * sin_phi) / tmp
    new_dy = dy * cos_theta + sin_theta * (dy * dz * cos_phi - dx * sin_phi) / tmp
    new_dz = dz * cos_theta - tmp * sin_theta * cos_phi

    # Normalize the new direction vector
    new_mag = np.sqrt((new_dx * new_dx) + (new_dy * new_dy) + (new_dz * new_dz))

    new_dx = new_dx / new_mag
    new_dy = new_dy / new_mag
    new_dz = new_dz / new_mag

    return geo.Vector3D.from_array(np.array([new_dx, new_dy, new_dz]))


def sample_U01() -> np.float32:
    """Returns a value uniformly sampled from the interval [0,1]"""
    # TODO: implement RANECU? Could be useful for validating translation to CUDA
    return np.random.random_sample()


def sample_initial_energy(spectrum: np.ndarray) -> np.float32:
    """Determine the energy (in eV) of a photon emitted by an X-Ray source with the given spectrum

    Args:
        spectrum (np.ndarray): the data associated with the spectrum.  Cross-reference spectral_data.py
    
    Returns:
        np.float32: the energy of a photon, in eV
    """
    total_count = sum(spectrum[:, 1])
    threshold = sample_U01() * total_count
    accumulator = 0
    for i in range(spectrum.shape[0] - 1):
        accumulator = accumulator + spectrum[i, 1]
        if accumulator >= threshold:
            return spectrum[i, 0]

    # If threshold hasn't been reached yet, we must have sampled the highest energy level
    return spectrum[-1, 0]


def sample_Rayleigh_theta(
    photon_energy: np.float32, rayleigh_sampler: RITA
) -> np.float32:
    """Randomly sample values of theta for a given Rayleigh scatter interaction
    Based on page 49 of paper 'PENELOPE-2006: A Code System for Monte Carlo Simulation of Electron and Photon Transport'

    Note that the materials files distributed with MC-GPU_v1.3 (https://code.google.com/archive/p/mcgpu/downloads) uses
    Form Factor data from PENELOPE-2006 files.  Accordingly, the (unnormalized) PDF is factored as:
            p_{Ra}(\\cos \\theta) = g(\\cos \\theta)[F(x,Z)]^2
    not
            p_{Ra}(\\cos \\theta) = g(\\cos \\theta)[F(q,Z)]^2
    Accordingly, we compute cos(theta) using the x-values, not the q-values

    Args:
        photon_energy (np.float32): the energy of the incoming photon
        rayleigh_sampler (RITA): the RITA sampler object for the material at the location of the interaction

    Returns:
        np.float32: cos(theta), where theta is the polar scattering angle 
    """
    kappa = photon_energy / ELECTRON_REST_ENERGY
    # Sample a random value of x^2 from the distribution pi(x^2), restricted to the interval (0, x_{max}^2)
    x_max = 20.6074 * 2 * kappa
    x_max2 = x_max * x_max
    x2 = rayleigh_sampler.sample_rita()
    while x2 > x_max2:
        # Resample until x^2 is in the interval (0, x_{max}^2)
        x2 = rayleigh_sampler.sample_rita()

    while True:
        # Set cos_theta
        cos_theta = 1 - (2 * x2 / x_max2)

        # Test cost_theta
        g = (1 + cos_theta * cos_theta) / 2

        if sample_U01() <= g:
            break

    return cos_theta


def sample_Compton_theta_E_prime(
    photon_energy: np.float32, mat_nshells: np.int32, mat_compton_data: np.ndarray
) -> np.float32:
    """Randomly sample values of theta and W for a given Compton scatter interaction

    Args:
        photon_energy (np.float32): the energy of the incoming photon
        mat_nshells (np.int32): the number of electron shells in the material being interacted with
        mat_compton_data (np.ndarray): the Compton scatter data for the material being interacted with.  See mcgpu_compton_data.py for more details 

    Returns:
        np.float32: cos_theta, the polar scattering angle 
        np.float32: E_prime, the energy of the outgoing photon
    """
    kappa = photon_energy / ELECTRON_REST_ENERGY

    a_1 = np.log1p(2 * kappa)
    one_p2k = 1 + 2 * kappa
    a_2 = 2 * kappa * (1 + kappa) / (one_p2k * one_p2k)

    tau_min = 1 / one_p2k

    ### Sample cos_theta

    # Compute S(E, \theta=\pi) here, since it does not depend on cos_theta
    s_pi = 0
    for shell in range(mat_nshells):
        U_i = mat_compton_data[shell, 1]
        if photon_energy > U_i:  # this serves as the Heaviside function
            left_term = (
                photon_energy * (photon_energy - U_i) * 2
            )  # since (1 - \cos(\theta=\pi)) == 2
            p_i_max = (left_term - ELECTRON_REST_ENERGY * U_i) / (
                ELECTRON_REST_ENERGY * np.sqrt(2 * left_term + U_i * U_i)
            )

            # Use several steps to calculate n_{i}(p_{i,max})
            tmp = mat_compton_data[shell, 2] * p_i_max  # J_{i,0} p_{i,max}
            tmp = (1 - tmp - tmp) if (p_i_max < 0) else (1 + tmp + tmp)
            exponent = 0.5 - 0.5 * tmp * tmp
            tmp = 0.5 * np.exp(exponent)
            if p_i_max > 0:
                tmp = 1 - tmp
            # 'tmp' now holds n_{i}(p_{i,max})

            s_pi = (
                s_pi + mat_compton_data[shell, 0] * tmp
            )  # Equivalent to: s_pi += f_{i} n_{i}(p_{i,max})
    # s_pi is now set

    cos_theta = None
    # local storage for the results of calculating n_{i}(p_{i,max})
    n_p_i_max_vals = [0 for i in range(COMPTON_MAX_NSHELLS)]

    while True:  # emulate do-while loop
        i = (
            1 if sample_U01() < (a_1 / (a_1 + a_2)) else 2
        )  # in CUDA code, we will be able to avoid using a variable to store i
        trnd = sample_U01()  # random number for calculating tau
        tau = (
            np.power(tau_min, trnd)
            if (1 == i)
            else np.sqrt(trnd + tau_min * tau_min * (1 - trnd))
        )
        cos_theta = 1 - (1 - tau) / (kappa * tau)

        # Compute S(E, \theta)
        s_theta = 0
        one_minus_cos = 1 - cos_theta
        for shell in range(mat_nshells):
            U_i = mat_compton_data[shell, 1]
            if photon_energy > U_i:  # this serves as the Heaviside function
                left_term = photon_energy * (photon_energy - U_i) * one_minus_cos
                p_i_max = (left_term - ELECTRON_REST_ENERGY * U_i) / (
                    ELECTRON_REST_ENERGY * np.sqrt(2 * left_term + U_i * U_i)
                )

                # Use several steps to calculate n_{i}(p_{i,max})
                tmp = mat_compton_data[shell, 2] * p_i_max  # J_{i,0} p_{i,max}
                tmp = (1 - tmp - tmp) if (p_i_max < 0) else (1 + tmp + tmp)
                exponent = 0.5 - 0.5 * tmp * tmp
                tmp = 0.5 * np.exp(exponent)
                if p_i_max > 0:
                    tmp = 1 - tmp
                # 'tmp' now holds n_{i}(p_{i,max})

                n_p_i_max_vals[shell] = tmp  # for later use in sampling E_prime

                s_theta = (
                    s_theta + mat_compton_data[shell, 0] * tmp
                )  # Equivalent to: s_pi += f_{i} n_{i}(p_{i,max})
            else:
                n_p_i_max_vals[shell] = 0

        # s_theta is now set

        # Compute the term of T(cos_theta) that does not involve S(E,\theta)
        T_tau_term = 1 - ((1 - tau) * ((2 * kappa + 1) * tau - 1)) / (
            kappa * kappa * tau * (1 + tau * tau)
        )

        # Test for acceptance
        if (s_pi * sample_U01()) <= (T_tau_term * s_theta):
            break

    # cos_theta is set by now

    # Choose the active shell
    p_z_omc = None  # p_z / (m_{e} c)

    while True:  # emulate do-while loop
        #
        # Steps:
        #   1. Choose a threshold value in range [0, s_theta]
        #   2. Accumulate the partial sum of f_{i} \Theta(E - U_i) n_{i}(p_{i,max}) over the electron shells
        #   3. Once the partial sum reaches the threshold value, we 'return' the most recently considered
        #       shell. In this manner, we select the active electron shell with relative probability equal
        #       to f_{i} \Theta(E - U_i) n_{i}(p_{i,max}).
        #   4. Calculate a random value of p_z
        #   5. Reject p_z and start over if p_z < -1 * m_{e} * c
        #   6. Calculate F_{max} and F_{p_z} and reject appropriately
        #
        threshold = sample_U01() * s_theta
        accumulator = 0
        active_shell = None
        for shell in range(mat_nshells):
            accumulator += mat_compton_data[shell, 0] * n_p_i_max_vals[shell]
            if accumulator >= threshold:
                active_shell = shell
                break
        # active_shell is now set

        two_A = sample_U01() * 2 * n_p_i_max_vals[active_shell]
        if two_A < 1:
            p_z_omc = 0.5 - np.sqrt(0.25 - 0.5 * np.log(two_A))
        else:
            p_z_omc = np.sqrt(0.25 - 0.5 * np.log(2 - two_A)) - 0.5
        p_z_omc = (
            p_z_omc / mat_compton_data[active_shell, 2]
        )  # Equivalent to: p_z_omc = p_z_omc / (J_{i,0} m_{e} c), completing the calculation

        if p_z_omc < -1:
            continue

        # Calculate F(p_z), where p_z is the PENELOPE-2006 'p_z' divided by (m_{e} c)
        beta2 = (
            1 + (tau * tau) - (2 * tau * cos_theta)
        )  # beta2 = (\beta)^2, where \beta := (c q_{C}) / E
        beta_tau_factor = np.sqrt(beta2) * (1 + tau * (tau - cos_theta) / beta2)
        F_p_z = 1 + beta_tau_factor * p_z_omc
        F_max = 1 + beta_tau_factor * (0.2 * (-1 if p_z_omc < 0 else 1))
        # NOTE: when converting to CUDA, I will want to see what happens when I "multiply everything through" by beta2.
        # That way, when comparing F_p_z with (\xi * F_max), there will only be multiplications and no divisions

        if sample_U01() * F_max < F_p_z:
            break  # p_z is accepted

    # p_z_omc is now set. Calculate E_ratio = E_prime / E
    t = p_z_omc * p_z_omc
    term_tau = 1 - t * tau * tau
    term_cos = 1 - t * tau * cos_theta

    tmp = np.sqrt(term_cos * term_cos - term_tau * (1 - t))
    if p_z_omc < 0:
        tmp = -1 * tmp
    tmp = term_cos + tmp

    E_ratio = tau * tmp / term_tau

    return cos_theta, E_ratio * photon_energy
