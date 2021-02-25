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

from .mcgpu_mfp_data import mfp_data, mfp_woodcock
from .mcgpu_compton_data import MAX_NSHELLS as COMPTON_MAX_NSHELLS
from .mcgpu_compton_data import material_nshells, compton_data

from .mcgpu_rita_samplers import rita_samplers

import math # for 'count_milestones'
import time # for keeping track of how long things take


logger = logging.getLogger(__name__)


#
# USEFUL CONSTANTS
#

LIGHT_C_METERS = 2.99792458e08 # m/s
ELECTRON_MASS_KILOS = 9.1093826e-31 # kg

LIGHT_C = 2.99792458e10 # cm/s
ELECTRON_MASS = 9.1093826e-28 # g

ELECTRON_REST_ENERGY = 510998.918 # eV


def simulate_scatter_no_vr(
    volume: vol.Volume,
    source_ijk: geo.Point3D,
    ijk_from_index: np.ndarray,
    index_from_ijk: np.ndarray,
    sensor_size: Tuple[int, int],
    output_shape: Tuple[int, int],
    spectrum: Optional[np.ndarray] = spectral_data.spectrums['90KV_AL40'], 
    photon_count: Optional[int] = 10000000, # 10^7
    E_abs: Optional[np.float32] = 5000
) -> np.ndarray:
    """Produce a grayscale (intensity-based) image representing the photon scatter during an X-Ray, 
    without using VR (variance reduction) techniques.

    Args:
        volume (np.ndarray): the volume density data.
        source (geo.Point3D): the source point for rays in the camera's IJK space
        ijk_from_index (np.ndarray): the ray transform for the projection.  Transforms pixel indices (u,v,1) to IJK offset from the X-Ray source
        index_from_ijk (np.ndarray): the inverse transformation of ijk_from_index
        sensor_size (Tuple[int,int]): the sensor size {width}x{height}, in pixels, of the detector
        output_shape (Tuple[int, int]): the {height}x{width} dimensions of the output image
        spectrum (Optional[np.ndarray], optional): spectrum array.  Defaults to 90KV_AL40 spectrum.
        photon_count (Optional[int], optional): the number of photons simulated.  Defaults to 10^7 photons.
        E_abs (Optional[np.float32], optional): the energy (in eV) at or below which photons are assumed to be absorbed by the materials.  Defaults to 1000 (eV).

    Returns:
        np.ndarray: intensity image of the photon scatter
    """
    count_milestones = [int(math.pow(10, i)) for i in range(int(1 + math.ceil(math.log10(photon_count))))] # [1, 10, 100, ..., 10^7] in default case

    accumulator = np.zeros(output_shape).astype(np.float32)

    material_ids = {}
    for i, mat_name in enumerate(volume.materials.keys()):
        material_ids[i] = mat_name
    assert len(material_ids) > 0
    
    # Convert the volume segmentation data from one-hot to [0..N-1]-labeled
    assert np.all(np.equal(volume.data.shape, volume.materials[material_ids[0]].shape))
    labeled_seg = np.empty_like(volume.data)

    for ilabel in material_ids.keys():
        labeled_seg = np.add(labeled_seg, ilabel * volume.materials[material_ids[ilabel]])

    # Plane data
    vol_planes = get_volume_surface_planes(volume)
    detector_plane = get_detector_plane(ijk_from_index, source_ijk, sensor_size)
    print(f"detector plane:")
    print(f"\t{detector_plane.plane_vector}")
    print(f"\t{detector_plane.surface_origin}")
    print(f"\t{detector_plane.basis_1}")
    print(f"\t{detector_plane.basis_2}")

    ###
    print(f"volume shape: {volume.data.shape}")
    print(f"source_ijk: {source_ijk}")
    print(f"Start time: {time.asctime()}")
    ###

    photon_hits = 0 # counts the number of photons that hit the detector
    for i in range(photon_count):
        if (i+1) in count_milestones:
            print(f"Simulating photon history {i+1} / {photon_count}")
            print(f"\tCurrent time: {time.asctime()}")
        initial_dir, initial_pos, sample_count = sample_initial_direction(vol_planes, source_ijk)
        initial_E = sample_initial_energy(spectrum)
        pixel_x, pixel_y, energy, num_scatter_events = track_single_photon_no_vr(
            initial_pos, 
            initial_dir, 
            initial_E, 
            E_abs, 
            labeled_seg, 
            volume.data, 
            detector_plane, 
            index_from_ijk, 
            material_ids
        )

        # Only keep track of photons that were scattered
        if 0 == num_scatter_events:
            continue
        else:
            print(f"photon history {i+1} / {photon_count}: {num_scatter_events} scatter events")
        
        # Model for detector: ideal image formation
        # Each pixel counts the total energy of the X-rays that enter the pixel (100% efficient pixels)
        # NOTE: will have to take care that this sort of tallying is <<<ATOMIC>>> in the GPU
        if (pixel_x < 0) or (pixel_x >= sensor_size[0]):
            continue
        if (pixel_y < 0) or (pixel_y >= sensor_size[1]):
            continue
        accumulator[pixel_x, pixel_y] = accumulator[pixel_x, pixel_y] + energy
        photon_hits += 1
    
    print(f"Finished simulating {photon_count} photon histories.  {photon_hits} / {photon_count} photons hit the detector")
    return accumulator

def track_single_photon_no_vr(
    initial_pos: geo.Point3D,
    initial_dir: geo.Vector3D,
    initial_E: np.float32,
    E_abs: np.float32,
    labeled_seg: np.ndarray,
    density_vol: np.ndarray,
    detector_plane: PlaneSurface,
    index_from_ijk: np.ndarray,
    material_ids: Dict[int, str]
) -> Tuple[int,int, np.float32, int]:
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

    print(f"initial_pos: {initial_pos}")
    print(f"initial_dir: {initial_dir}")


    photon_energy = initial_E # tracker variable throughout the duration of the photon history

    num_scatter_events = 0

    while True: # emulate a do-while loop
        # Get voxel (index) coord.s.  Keep in mind that IJK coord.s are voxel-centered
        vox_x = int(np.floor(pos.data[0]))
        vox_y = int(np.floor(pos.data[1]))
        vox_z = int(np.floor(pos.data[2]))
        if (vox_x < 0) or (vox_x >= density_vol.shape[0]):
            break
        if (vox_y < 0) or (vox_y >= density_vol.shape[1]):
            break
        if (vox_z < 0) or (vox_z >= density_vol.shape[2]):
            break
        #voxel_coords = (vox_x, vox_y, vox_z)
        print(f"INSIDE VOLUME")
        mat_label = labeled_seg[vox_x, vox_y, vox_z]
        mat_name = material_ids[mat_label]

        mfp_wc = get_woodcock_mfp(photon_energy)
        mfp_Ra, mfp_Co, mfp_Tot = get_mfp_data(mfp_data[mat_name], photon_energy)

        # Delta interactions
        while True:
            # simulate moving the photon
            s = -1 * (10 * mfp_wc) * np.log(sample_U01()) # multiply by 10 to convert from MFP data (cm) to voxel spacing (mm)
            pos = pos + (s * direction)

            # Check for leaving the volume
            vox_x = int(np.floor(pos.data[0]))
            vox_y = int(np.floor(pos.data[1]))
            vox_z = int(np.floor(pos.data[2]))
            if (vox_x < 0) or (vox_x >= density_vol.shape[0]):
                break
            if (vox_y < 0) or (vox_y >= density_vol.shape[1]):
                break
            if (vox_z < 0) or (vox_z >= density_vol.shape[2]):
                break

            mat_label = labeled_seg[vox_x, vox_y, vox_z]
            mat_name = material_ids[mat_label]
            
            mfp_Ra, mfp_Co, mfp_Tot = get_mfp_data(mfp_data[mat_name], photon_energy)

            print(f"probability to accept the collision: mfp_wc / mfp_Tot == {mfp_wc / mfp_Tot}")

            if sample_U01() < mfp_wc / mfp_Tot:
                # Accept the collision.  See http://serpent.vtt.fi/mediawiki/index.php/Delta-_and_surface-tracking
                break
            print(f"DELTA COLLISION")
        
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
            cos_theta, E_prime = sample_Compton_theta_E_prime(photon_energy, material_nshells[mat_name], compton_data[mat_name])
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
        print(f"SCATTER EVENT")

        phi = 2 * np.pi * sample_U01()
        direction = get_scattered_dir(direction, cos_theta, phi)

        # END WHILE
    
    # final processing

    # Transport the photon to the detector plane
    hits_detector_dist = detector_plane.check_ray_intersection(pos, direction)
    if hits_detector_dist is None:
        return -1, -1, photon_energy, num_scatter_events
    
    hit = geo.Point3D.from_any(pos + (hits_detector_dist * direction))

    #print(f"hit: {hit}")

    # NOTE: an alternative formulation would be to use (rt_kinv).inv
    #pixel_x, pixel_y = detector_plane.get_lin_comb_coefs(hit)
    #print(f"old pixel: {pixel_x}, {pixel_y}")
    
    hit_x = hit.data[0]
    hit_y = hit.data[1]
    hit_z = hit.data[2]
    pixel_x = index_from_ijk[0,0] * hit_x + index_from_ijk[0,1] * hit_y + index_from_ijk[0,2] * hit_z + index_from_ijk[0,3]
    pixel_y = index_from_ijk[1,0] * hit_x + index_from_ijk[1,1] * hit_y + index_from_ijk[1,2] * hit_z + index_from_ijk[1,3]

    #print(f"new pixel: {pixel_x}, {pixel_y}")
    
    return int(np.floor(pixel_x)), int(np.floor(pixel_y)), photon_energy, num_scatter_events

def get_mfp_data(
    table: np.ndarray,
    E: np.float32
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
    lo_idx = 0 # inclusive
    hi_idx = table.shape[0] # exclusive
    i = None # the index of the bin that we find E in

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

def get_woodcock_mfp(
    E: np.float32
) -> np.float32:
    """Access the Woodcock Mean Free Path at the given photon energy level.  
    For an explanation of what the Woodcock Mean Free Path is, see mcgpu_mfp_data.py.
    Performs linear interpolation for any energy value that isn't exactly a table entry.

    Args:
        E (np.float32): the energy of the photon
    
    Returns:
        np.float32: the inverse of the total majorant cross section.  This returned value has units of centimeters.
    """
    # Binary search to find the proper table entry.  Want energy(lo_bin) <= E < energy(hi_bin), with (lo_bin + 1) == hi_bin
    lo_idx = 0 # inclusive
    hi_idx = mfp_woodcock.shape[0] # exclusive
    i = None # the index of the bin that we find E in

    while lo_idx < hi_idx:
        mid_idx = np.floor_divide(lo_idx + hi_idx, np.int32(2))

        if E < mfp_woodcock[mid_idx, 0]:
            # Need to check lower intervals
            hi_idx = mid_idx
        elif E < mfp_woodcock[mid_idx + 1, 0]:
            # found correct interval
            i = mid_idx
            break
        else:
            # Need to check higher intervals
            lo_idx = mid_idx + 1
    
    assert (mfp_woodcock[i, 0] <= E) and (E < mfp_woodcock[i + 1, 0])
    
    # Linear interpolation 
    delta_E = mfp_woodcock[i + 1, 0] - mfp_woodcock[i, 0]
    partial = E - mfp_woodcock[i, 0]

    delta_mfp_Tot = mfp_woodcock[i + 1, 1] - mfp_woodcock[i, 1]

    mfp_wc = mfp_woodcock[i, 1] + (delta_mfp_Tot * partial) / delta_E

    return mfp_wc

def get_volume_surface_planes(
    volume: vol.Volume
) -> List[PlaneSurface]:
    """Given a volume, returns the PlaneSurface objects for each of the six planes on the surface of the rectangular prism volume in a raw Python array.

    Everything is in IJK coordinates.  In this geometry, 3D coordinates (0,0,0) is the center of the voxel referenced by indexing into the volume at [0,0,0].

    Args:
        volume (vol.Volume): the volume in question.
    
    Returns:
        List[PlaneSurface]: the 6 PlaneSurface objects, where the i-th object corresponds to the i-th face of the volume (no particular ordering)
    """
    # Since the volume is a rectangular prism, each of the normal vectors are going to be (+/-)1 in a single direction.
    # The "distance from the origin" is then determined by which face is under consideration
    # NOTE: as a stylistic choice, I am choosing the normal vectors to point outward
    x_len = volume.data.shape[0]
    y_len = volume.data.shape[1]
    z_len = volume.data.shape[2]

    plane_vectors = [
        np.array([-1, 0, 0, 0.5]),
        np.array([1, 0, 0, x_len - 0.5]),
        np.array([0, -1, 0, 0.5]),
        np.array([0, 1, 0, y_len - 0.5]),
        np.array([0, 0, -1, 0.5]),
        np.array([0, 0, 1, z_len - 0.5])
    ]
    surface_origins = [
        geo.Point3D.from_array(np.array([-0.5, 0, 0])),
        geo.Point3D.from_array(np.array([x_len - 0.5, 0, 0])),
        geo.Point3D.from_array(np.array([0, -0.5, 0])),
        geo.Point3D.from_array(np.array([0, y_len - 0.5, 0])),
        geo.Point3D.from_array(np.array([0, 0, -0.5])),
        geo.Point3D.from_array(np.array([0, 0, z_len - 0.5]))
    ]

    x_dir = geo.Vector3D.from_array(np.array([1, 0, 0]))
    y_dir = geo.Vector3D.from_array(np.array([0, 1, 0]))
    z_dir = geo.Vector3D.from_array(np.array([0, 0, 1]))
    bases = [
        (geo.Vector3D.from_any(y_dir), geo.Vector3D(z_dir)),
        (geo.Vector3D.from_any(y_dir), geo.Vector3D(z_dir)),
        
        (geo.Vector3D.from_any(x_dir), geo.Vector3D(z_dir)),
        (geo.Vector3D.from_any(x_dir), geo.Vector3D(z_dir)),

        (geo.Vector3D.from_any(x_dir), geo.Vector3D(y_dir)),
        (geo.Vector3D.from_any(x_dir), geo.Vector3D(y_dir))
    ]
    bounds = [
        np.array([[-0.5, y_len - 0.5], [-0.5, z_len - 0.5]]), 
        np.array([[-0.5, y_len - 0.5], [-0.5, z_len - 0.5]]), 

        np.array([[-0.5, x_len - 0.5], [-0.5, z_len - 0.5]]),
        np.array([[-0.5, x_len - 0.5], [-0.5, z_len - 0.5]]),

        np.array([[-0.5, x_len - 0.5], [-0.5, y_len - 0.5]]),
        np.array([[-0.5, x_len - 0.5], [-0.5, y_len - 0.5]]),
    ]

    return [PlaneSurface(plane_vectors[i], surface_origins[i], bases[i], bounds[i], True) for i in range(6)]

def sample_initial_direction(
    surfaces: List[PlaneSurface],
    source_ijk: geo.Point3D
) -> Tuple[geo.Vector3D, geo.Point3D, int]:
    """Returns an initial direction vector for a photon that is guaranteed to hit the volume, as well as the coordinates of that initial intersection with the volume.
    Behaves by randomly sampling \\theta from [0, PI] and \\phi from [0, 2 * PI], then determining if a photon going in that direction will hit the volume.
    
    Args:
        surfaces (List[PlaneSurface]): six PlaneSurface objects, one for each face of the volume
        source_ijk (geo.point3D): the IJK coordinates of the X-Ray source, relative to the IJK origin (indices [0,0,0] in the volume)

    Returns:
        geo.Vector3D: the initial direction unit vector (dx, dy, dz)^T
        geo.Point3D: the location where the photon first enters the volume
        int: the number of times a direction was sampled before returning
    """
    sample_count = 0

    intersection_points = [] # since there could be multiple points that the ray intersects the volume
    distances = []
    direction = None

    while True:
        # Sampling explanation here: http://corysimon.github.io/articles/uniformdistn-on-sphere/
        phi = 2 * np.pi * sample_U01() # azimuthal angle
        theta = np.arccos(1 - 2 * sample_U01()) # polar angle

        sin_theta = np.sin(theta)
        
        dx = sin_theta * np.cos(phi)
        dy = sin_theta * np.sin(phi)
        dz = np.cos(theta)

        direction = geo.Vector3D.from_array(np.array([dx, dy, dz]))
        sample_count += 1

        intersection_points = [] # since there could be multiple points that the ray intersects the volume

        for i in range(6):
            distance = surfaces[i].check_ray_intersection(source_ijk, direction)
            if distance < 0:
                continue
            intersection = geo.Point3D.from_any(source_ijk + (distance * direction))
            
            if surfaces[i].point_on_surface(intersection):
                intersection_points.append(intersection)
                distances.append(distance)
        
        if 0 < len(intersection_points):
            break

        # Othewise, re-samples the direction

    assert len(intersection_points) == len(distances)

    # Return the intersection point that is closest to the X-Ray source
    min_dist_idx = 0
    min_dist = float("inf")
    for i in range(len(intersection_points)):
        if distances[i] < min_dist:
            min_dist = distances[i]
            min_dist_idx = i
    
    return direction, intersection_points[min_dist_idx], sample_count

def get_detector_plane(
    rt_kinv: np.ndarray,
    source_ijk: geo.Point3D,
    sensor_size: Tuple[int,int]
) -> PlaneSurface:
    """Calculates the PlaneSurface object of the detector plane in IJK coordinates.
    Note that the cosines of the plane's normal vector (n_x, n_y, n_z) are NOT normalized to be a unit vector.

    The first basis vector represents

    Args:
        rt_kinv (np.ndarray): the ray transform for the projection.  Transforms pixel indices (u,v,1) to IJK offset from the X-Ray source
        source_ijk (geo.Point3D): the IJK coordinates of the X-Ray source, relative to the IJK origin (indices [0,0,0] in the volume)
        sensor_size (Tuple[int,int]): the sensor size {width}x{height}, in pixels, of the detector

    Returns:
        PlaneSurface: a PlaneSurface object representing the detector.  
        np.ndarray: a basis vector that represents how increasing pixel x-coordinate by 1 affects the 3D position
        np.ndarray: a basis vector that represents how increasing pixel y-coordinate by 1 affects the 3D position
    """
    # Based off the project_kernel.cu code:
    #   Let \hat{p} = (u,v,1)^T be the pixel coord.s on the detector plane
    #   Then, the 3D IJK coord.s of that pixel are related to (R^T K^{-1}) \hat{p} == (rt_kinv) @ \hat{p}
    #   Specifically, (rt_kinv) @ \hat{p} is the IJK vector from the X-Ray source to the pixel (u,v) on 
    #   the detector plane.
    #
    # We calculate the normal vector of the detector plane in IJK by using the three-point method:
    #   1. Let {p1, p2, p3} be three pixel coordinates of the form (u, v, 1)^T
    #   2. Three coplanar points in IJK coordinates are r1 := (rt_kinv) @ p1, r2 := (rt_kinv) @ p2, 
    #      r3 := (rt_kinv) @ p3
    #   3. Compute two vectors that are -in- the plane: v1 := r2 - r1, v2 := r3 - r1
    #   4. The cross product v := v1 x v2 is perpendicular to both v1 and v2.  Thus, v is a normal vector to the plane
    #
    # Note that even though {r1, r2, r3} are technically the vectors points from the X-ray source to the detector plane,
    # not pointing from the IJK origin to the detector plane, the fact that {v1, v2} are [relative displacement vectors]
    # means that the shift in "origin" for {r1, r2, r3} has no effect on calculating the normal vector for the detector plane.
    #
    # Simplifying the math to reduce the number of arithmetic steps:
    #   v1 = r2 - r1 = (rt_kinv) @ p2 - (rt_kinv) @ p1 = (rt_kinv) @ (p2 - p1)
    #   v2 = r3 - r1 = (rt_kinv) @ p3 - (rt_kinv) @ p1 = (rt_kinv) @ (p3 - p1)
    #
    # Choosing easy p_i's of: p1 = (0, 0, 1)^T, p2 = (1, 0, 1)^T, p3 = (0, 1, 1)^T, we get:
    #   v1 = (rt_kinv) @ (1, 0, 0)^T = [first column of rt_kinv]        // corresponds to moving 1 pixel over in x-direction (x increases)  
    #   v2 = (rt_kinv) @ (0, 1, 0)^T = [second column of rt_kinv]       // corresponds to moving 1 pixel down in y-direction (y increases)
    #
    # To reduce the number of characters, let M: = (rt_kinv), as a 9-element row-major ordering of the 3x3 (rt_kinv).
    #   v := v1 x v2 = (M[0], M[3], M[6])^T x (M[1], M[4], M[7])^T
    #       = (
    #           M[3] * M[7] - M[6] * M[4],
    #           M[6] * M[1] - M[0] * M[7],
    #           M[0] * M[4] - M[3] * M[1]  
    #         )^T
    #       = (
    #           rt_kinv[1,0] * rt_kinv[2,1] - rt_kinv[2,0] * rt_kinv[1,1],
    #           rt_kinv[2,0] * rt_kinv[0,1] - rt_kinv[0,0] * rt_kinv[2,1],
    #           rt_kinv[0,0] * rt_kinv[1,1] - rt_kinv[1,0] * rt_kinv[0,1]
    #         )^T
    #
    # Once we have the normal vector, we need the minimum distance between the detector plane and 
    # the origin of the IJK coord.s to get the fourth entry in the 'plane vector' (n_x, n_y, n_z, d)

    print("rt_kinv:")
    print(f"\t[{rt_kinv[0,0]} {rt_kinv[0,1]} {rt_kinv[0,2]}]")
    print(f"\t[{rt_kinv[1,0]} {rt_kinv[1,1]} {rt_kinv[1,2]}]")
    print(f"\t[{rt_kinv[2,0]} {rt_kinv[2,1]} {rt_kinv[2,2]}]")
    
    # Normal vector to the detector plane:
    vx = rt_kinv[1,0] * rt_kinv[2,1] - rt_kinv[2,0] * rt_kinv[1,1]
    vy = rt_kinv[2,0] * rt_kinv[0,1] - rt_kinv[0,0] * rt_kinv[2,1]
    vz = rt_kinv[0,0] * rt_kinv[1,1] - rt_kinv[1,0] * rt_kinv[0,1]

    v_mag = np.sqrt((vx * vx) + (vy * vy) + (vz * vz))

    vx /= v_mag
    vy /= v_mag
    vz /= v_mag

    print(f"vx, vy, vz: {vx}, {vy}, {vz}")

    # Distance from the detector plane to the origin of the IJK coordinates
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
    # Vector SC can be found by finding the IJK coordinates of the detector center (vector OC) and 
    # the IJK coordinates of the X-ray source (vector OS).  SC = OC - OS
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

    cu = sensor_size[0] / 2 # pixel coord.s of the center of the detector
    cv = sensor_size[1] / 2

    # IJK coord.s of the SC, the ray from the X-Ray source S to the center C of the detector
    sc_x = cu * rt_kinv[0,0] + cv * rt_kinv[0,1] + rt_kinv[0,2]
    sc_y = cu * rt_kinv[1,0] + cv * rt_kinv[1,1] + rt_kinv[1,2]
    sc_z = cu * rt_kinv[2,0] + cv * rt_kinv[2,1] + rt_kinv[2,2]

    # Note that the IJK coord.s of vector OS are contained in source_ijk, which is obtained 
    # by calling the camera_center_in_volume method
    sc_dot_sc = (sc_x * sc_x) + (sc_y * sc_y) + (sc_z * sc_z)
    sc_dot_os = (sc_x * source_ijk.data[0]) + (sc_y * source_ijk.data[1]) + (sc_z * source_ijk.data[2])

    d = np.sqrt(sc_dot_sc) * (1 + (sc_dot_os / sc_dot_sc))

    plane_vector = np.array([vx, vy, vz, np.abs(d)])

    # The 'surface origin' corresponds to the pixel [0,0] on the detector.
    # Vector source_to_surf_ori = (rt_kinv) @ (0,0,1)^T = [third column of rt_kinv]
    surf_ori_x = rt_kinv[0,2] + source_ijk.data[0] # source_to_surf_ori + origin_to_source == origin_to_surf_ori
    surf_ori_y = rt_kinv[1,2] + source_ijk.data[1]
    surf_ori_z = rt_kinv[2,2] + source_ijk.data[2]
    # SANITY CHECK: after using an inverse-of-upper-triangular-matrix formula, we get:
    # kinv[2] == (s c_y - c_x f_y) / (f_x f_y)
    # kinv[5] == c_y / f_y
    # kinv[8] == 1
    surface_origin = geo.Point3D.from_array(np.array([surf_ori_x, surf_ori_y, surf_ori_z]))

    # The basis is {v1, v2}, where {v1, v2} are the vectors described in the "choosing easy p_i's" section
    # That way, the point of intersection is:
    #
    #   intersection = surface_origin + (pixel_x_value) * v1 + (pixel_y_value) * v2
    #
    v1 = geo.Vector3D.from_array(np.array([rt_kinv[0,0], rt_kinv[1,0], rt_kinv[2,0]]))
    v2 = geo.Vector3D.from_array(np.array([rt_kinv[0,1], rt_kinv[1,1], rt_kinv[2,1]]))

    # Coordinate bounds correspond to the size of the detector, in pixels
    bounds = np.array([[0, sensor_size[0]],
                       [0, sensor_size[1]]])

    # Not guaranteed that the basis vectors are orthogonal
    return PlaneSurface(plane_vector, surface_origin, (v1, v2), bounds, False)

def get_scattered_dir(
    direction: geo.Vector3D,
    cos_theta: np.float32,
    phi: np.float32
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

def sample_initial_energy(
    spectrum: np.ndarray
) -> np.float32:
    """Determine the energy (in eV) of a photon emitted by an X-Ray source with the given spectrum

    Args:
        spectrum (np.ndarray): the data associated with the spectrum.  Cross-reference spectral_data.py
    
    Returns:
        np.float32: the energy of a photon, in eV
    """
    total_count = sum(spectrum[:,1])
    threshold = sample_U01() * total_count
    accumulator = 0
    for i in range(spectrum.shape[0] - 1):
        accumulator = accumulator + spectrum[i, 1]
        if accumulator >= threshold:
            return spectrum[i, 0]
    
    # If threshold hasn't been reached yet, we must have sampled the highest energy level
    return spectrum[-1, 0]

def sample_Rayleigh_theta(
    photon_energy: np.float32,
    rayleigh_sampler: RITA
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
    while (x2 > x_max2):
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
    photon_energy: np.float32,
    mat_nshells: np.int32,
    mat_compton_data: np.ndarray
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
        if photon_energy > U_i: # this serves as the Heaviside function
            left_term = photon_energy * (photon_energy - U_i) * 2 # since (1 - \cos(\theta=\pi)) == 2
            p_i_max = (left_term - ELECTRON_REST_ENERGY * U_i) / (ELECTRON_REST_ENERGY * np.sqrt(2 * left_term + U_i * U_i))
            
            # Use several steps to calculate n_{i}(p_{i,max})
            tmp = mat_compton_data[shell, 2] * p_i_max # J_{i,0} p_{i,max}
            tmp = (1 - tmp - tmp) if (p_i_max < 0) else (1 + tmp + tmp)
            exponent = 0.5 - 0.5 * tmp * tmp
            tmp = 0.5 * np.exp(exponent)
            if (p_i_max > 0):
                tmp = 1 - tmp
            # 'tmp' now holds n_{i}(p_{i,max})

            s_pi = s_pi + mat_compton_data[shell, 0] * tmp # Equivalent to: s_pi += f_{i} n_{i}(p_{i,max})
    # s_pi is now set

    cos_theta = None
    # local storage for the results of calculating n_{i}(p_{i,max})
    n_p_i_max_vals = [0 for i in range(COMPTON_MAX_NSHELLS)]

    while True: # emulate do-while loop
        i = 1 if sample_U01() < (a_1 / (a_1 + a_2)) else 2 # in CUDA code, we will be able to avoid using a variable to store i
        trnd = sample_U01() # random number for calculating tau
        tau = np.power(tau_min, trnd) if (1 == i) else np.sqrt(trnd + tau_min * tau_min * (1 - trnd))
        cos_theta = 1 - (1 - tau) / (kappa * tau)

        # Compute S(E, \theta)
        s_theta = 0
        one_minus_cos = 1 - cos_theta
        for shell in range(mat_nshells):
            U_i = mat_compton_data[shell, 1]
            if photon_energy > U_i: # this serves as the Heaviside function
                left_term = photon_energy * (photon_energy - U_i) * one_minus_cos
                p_i_max = (left_term - ELECTRON_REST_ENERGY * U_i) / (ELECTRON_REST_ENERGY * np.sqrt(2 * left_term + U_i * U_i))
                
                # Use several steps to calculate n_{i}(p_{i,max})
                tmp = mat_compton_data[shell, 2] * p_i_max # J_{i,0} p_{i,max}
                tmp = (1 - tmp - tmp) if (p_i_max < 0) else (1 + tmp + tmp)
                exponent = 0.5 - 0.5 * tmp * tmp
                tmp = 0.5 * np.exp(exponent)
                if (p_i_max > 0):
                    tmp = 1 - tmp
                # 'tmp' now holds n_{i}(p_{i,max})

                n_p_i_max_vals[shell] = tmp # for later use in sampling E_prime

                s_theta = s_theta + mat_compton_data[shell, 0] * tmp # Equivalent to: s_pi += f_{i} n_{i}(p_{i,max})
            else:
                n_p_i_max_vals[shell] = 0

        # s_theta is now set

        # Compute the term of T(cos_theta) that does not involve S(E,\theta)
        T_tau_term = 1 - ((1 - tau) * ((2 * kappa + 1) * tau - 1)) / (kappa * kappa * tau * (1 + tau * tau))

        # Test for acceptance
        if (s_pi * sample_U01()) <= (T_tau_term * s_theta):
            break
    
    # cos_theta is set by now

    # Choose the active shell
    p_z_omc = None # p_z / (m_{e} c) 

    while True: # emulate do-while loop
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
            if (accumulator >= threshold):
                active_shell = shell
                break
        # active_shell is now set

        two_A = sample_U01() * 2 * n_p_i_max_vals[active_shell]
        if two_A < 1:
            p_z_omc = 0.5 - np.sqrt(0.25 - 0.5 * np.log(two_A))
        else:
            p_z_omc = np.sqrt(0.25 - 0.5 * np.log(2 - two_A)) - 0.5
        p_z_omc = p_z_omc / mat_compton_data[active_shell,2] # Equivalent to: p_z_omc = p_z_omc / (J_{i,0} m_{e} c), completing the calculation

        if p_z_omc < -1:
            continue
        
        # Calculate F(p_z), where p_z is the PENELOPE-2006 'p_z' divided by (m_{e} c)
        beta2 = 1 + (tau * tau) - (2 * tau * cos_theta) # beta2 = (\beta)^2, where \beta := (c q_{C}) / E
        beta_tau_factor = np.sqrt(beta2) * (1 + tau * (tau - cos_theta) / beta2)
        F_p_z = 1 + beta_tau_factor * p_z_omc
        F_max = 1 + beta_tau_factor * (0.2 * (-1 if p_z_omc < 0 else 1))
        # NOTE: when converting to CUDA, I will want to see what happens when I "multiply everything through" by beta2.
        # That way, when comparing F_p_z with (\xi * F_max), there will only be multiplications and no divisions

        if sample_U01() * F_max < F_p_z:
            break # p_z is accepted
    
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
