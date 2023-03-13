/*
 * Based on Sisniega et al. (2015), "High-fidelity artifact correction for
 * cone-beam CT imaging of the brain"
 */

#include "scatter_header.cu"

extern "C" {
__global__ void simulate_scatter(
    int detector_width, // size of detector in pixels
    int detector_height,
    int histories_for_thread,   // number of photons for -this- thread to track
    char *labeled_segmentation, // [0..NUM_MATERIALS-1]-labeled segmentation
    float sx,                   // coordinates of source in IJK
    float sy, // (not in a float3_t for ease of calling from Python wrapper)
    float sz,
    float sdd,                  // source-to-detector distance [mm]
    int volume_shape_x,         // integer size of the volume to avoid
    int volume_shape_y,         // floating-point errors with the
                                // gVolumeEdge{Min,Max}Point
    int volume_shape_z,         // and gVoxelElementSize math
    float gVolumeEdgeMinPointX, // bounds of the volume in IJK
    float gVolumeEdgeMinPointY, float gVolumeEdgeMinPointZ,
    float gVolumeEdgeMaxPointX, float gVolumeEdgeMaxPointY,
    float gVolumeEdgeMaxPointZ,
    float gVoxelElementSizeX, // voxel size in world coordinates
    float gVoxelElementSizeY, float gVoxelElementSizeZ,
    float *index_from_world, // (2, 4) array giving the inverse of the ray
                             // transform
    mat_mfp_data_t *mfp_data_arr, wc_mfp_data_t *woodcock_mfp,
    compton_data_t *compton_arr, rayleigh_data_t *rayleigh_arr,
    plane_surface_t *detector_plane,
    float *world_from_ijk,    // 3x4 transform
    float *ijk_from_world,    // 3x4 transform
    int n_bins,               // the number of spectral bins
    float *spectrum_energies, // 1-D array -- size is the n_bins. Units: [keV]
    float *spectrum_cdf, // 1-D array -- cumulative density function over the
                         // energies
    float E_abs, // the energy level below which photons are assumed to be
                 // absorbed [keV]
    int seed_input,
    float *deposited_energy, // the output.  Size is
                             // [detector_width]x[detector_height]
    int *
        num_scattered_hits, // number of scattered photons that hit the detector
                            // at each pixel. Same size as deposited_energy.
    int *num_unscattered_hits // number of unscattered photons that hit the
                              // detector at each pixel. Same size as
                              // deposited_energy.
) {
  rng_seed_t seed;
  int thread_id = threadIdx.x + (blockIdx.x * blockDim.x); // 1D block
  initialize_seed(thread_id, histories_for_thread, seed_input, &seed);

  int3_t volume_shape;
  volume_shape.x = volume_shape_x;
  volume_shape.y = volume_shape_y;
  volume_shape.z = volume_shape_z;

  float3_t gVolumeEdgeMinPoint;
  gVolumeEdgeMinPoint.x = gVolumeEdgeMinPointX;
  gVolumeEdgeMinPoint.y = gVolumeEdgeMinPointY;
  gVolumeEdgeMinPoint.z = gVolumeEdgeMinPointZ;

  float3_t gVolumeEdgeMaxPoint;
  gVolumeEdgeMaxPoint.x = gVolumeEdgeMaxPointX;
  gVolumeEdgeMaxPoint.y = gVolumeEdgeMaxPointY;
  gVolumeEdgeMaxPoint.z = gVolumeEdgeMaxPointZ;

  /*float3_t gVoxelElementSize; // TODO: remove for disuse
  gVoxelElementSize.x = gVoxelElementSizeX;
  gVoxelElementSize.y = gVoxelElementSizeY;
  gVoxelElementSize.z = gVoxelElementSizeZ; */

  if (0 == thread_id) {
    /*printf("volume_shape: {%d, %d, %d}\n", volume_shape.x, volume_shape.y,
    volume_shape.z); printf("gVolumeEdgeMinPoint: {%f, %f, %f}\n",
    gVolumeEdgeMinPoint.x, gVolumeEdgeMinPoint.y, gVolumeEdgeMinPoint.z);
    printf("gVolumeEdgeMaxPoint: {%f, %f, %f}\n", gVolumeEdgeMaxPoint.x,
    gVolumeEdgeMaxPoint.y, gVolumeEdgeMaxPoint.z); printf("source: {%f, %f,
    %f}\n", sx, sy, sz); printf( "index_from_world:\n\t[%f, %f, %f, %f]\n\t[%f,
    %f, %f, %f]\n", index_from_world[0], index_from_world[1],
    index_from_world[2], index_from_world[3], index_from_world[4],
    index_from_world[5], index_from_world[6], index_from_world[7]
    );*/
    /*printf(
        "detector_plane:\n"
        "\t.n={%f, %f, %f}, .d=%f\n"
        "\t.ori={%f, %f, %f}\n"
        "\t.b1={%f, %f, %f}\n"
        "\t.b2={%f, %f, %f}\n"
        "\t.bound1=[%f, %f]\n"
        "\t.bound2=[%f, %f]\n",
        detector_plane->n.x, detector_plane->n.y, detector_plane->n.z,
    detector_plane->d, detector_plane->ori.x, detector_plane->ori.y,
    detector_plane->ori.z, detector_plane->b1.x, detector_plane->b1.y,
    detector_plane->b1.z, detector_plane->b2.x, detector_plane->b2.y,
    detector_plane->b2.z, detector_plane->bound1.x, detector_plane->bound1.y,
        detector_plane->bound2.x, detector_plane->bound1.y
    );
    printf("WOODCOCK MFP: n_bins=%d\n", woodcock_mfp->n_bins);
    for (int i = 0; i < NUM_MATERIALS; i++) {
        printf(
            "MATERIAL MFP #%d: n_bins=%d\n"
            "MATERIAL RITA #%d: n_gridpts=%d\n"
            "MATERIAL COMPTON #%d: nshells=%d\n",
            i, mfp_data_arr[i].n_bins,
            i, rita_arr[i].n_gridpts,
            i, compton_arr[i].nshells
        );
    }*/

    /*printf("STRUCTURE SIZES:\n");
    printf("\tplane_surface_t: %llu\n", sizeof(plane_surface_t));
    printf("\trng_seed_t: %llu\n", sizeof(rng_seed_t));
    printf("\trayleigh_data_t: %llu\n", sizeof(rayleigh_data_t));
    printf("\tmat_mfp_data_t: %llu\n", sizeof(mat_mfp_data_t));
    printf("\twc_mfp_data_t: %llu\n", sizeof(wc_mfp_data_t));
    printf("\tcompton_data_t: %llu\n", sizeof(compton_data_t));*/

    /*for (int i = 0; i < NUM_MATERIALS; i++) {
        printf("RAYLEIGH DATA #%d: n_gridpts=%d. memloc: %llu\n", i,
    rayleigh_arr[i].n_gridpts, &rayleigh_arr[i]); printf("&rayleigh_arr[i].x[0]:
    %llu, &(...).y[0]: %llu\n", &rayleigh_arr[i].x[0], &rayleigh_arr[i].y[0]);
        printf("&rayleigh_arr[i].a[0]: %llu, &(...).b[0]: %llu\n",
    &rayleigh_arr[i].a[0], &rayleigh_arr[i].b[0]);
        printf("&rayleigh_arr[i].pmax[0]: %llu, &(...).pmax[MAX_MFP_BINS-1]:
    %llu\n", &rayleigh_arr[i].pmax[0], &rayleigh_arr[i].pmax[MAX_MFP_BINS - 1]);
    }*/

    /*for (int i = 0; i < NUM_MATERIALS; i++) {
        printf("MATERIAL MFP #%d: n_bins=%d\n", i, mfp_data_arr[i].n_bins);
        for (int b = 0; b < mfp_data_arr[i].n_bins; b++) {
            printf("\t[%8f, %8f, %8f, %8f]\n", mfp_data_arr[i].energy[b],
    mfp_data_arr[i].mfp_Ra[b], mfp_data_arr[i].mfp_Co[b],
    mfp_data_arr[i].mfp_Tot[b]);
        }
    }*/
  }
  // return; // TODO: remove this when done with reading kernel structure data

  int histories_printing = 0; // TODO: this is ugly

  if (histories_printing)
    printf("thread #%d has started tracking, seed={%d, %d}\n", thread_id,
           seed.x, seed.y);
  // printf("histories_for_thread: %d\n", histories_for_thread);

  for (; histories_for_thread > 0; histories_for_thread--) {
    if (histories_printing)
      printf("%d histories left in thread #%d\n", histories_for_thread,
             thread_id);
    float3_t pos;
    pos.x = sx;
    pos.y = sy;
    pos.z = sz;

    float3_t dir;
    sample_initial_dir_world(&dir, &seed);
    int is_hit;
    move_photon_to_volume(&pos, &dir, &is_hit, &gVolumeEdgeMinPoint,
                          &gVolumeEdgeMaxPoint);
    if (is_hit) {
      // printf("hit volume\n"); // many of these get printed out
      float energy =
          1000.f * sample_initial_energy(n_bins, spectrum_energies,
                                         spectrum_cdf, &seed); // [eV]
      // is_hit gets repurposed since we don't need it anymore for 'did the
      // photon hit the volume'
      int num_scatter_events = 0;
      track_photon(&pos, &dir, &energy, &is_hit, &num_scatter_events,
                   (1000.f * E_abs),
                   labeled_segmentation, // Pass in E_abs in [eV]
                   mfp_data_arr, woodcock_mfp, compton_arr, rayleigh_arr,
                   &volume_shape, &gVolumeEdgeMinPoint, &gVolumeEdgeMaxPoint,
                   detector_plane, world_from_ijk, ijk_from_world, &seed);

      if (is_hit) {
        if (num_scatter_events > 0) {
          // printf("scattered photon ended up here: {%f, %f, %f}\n", pos.x,
          // pos.y, pos.z);
        }
        ///////////////printf("hit detector\n");
        // 'pos' contains the IJK coord.s of collision with the detector.

        shift_point_frame_3x4_transform(&pos, world_from_ijk);
        printf("hit detector plane at world coord.s: (%f, %f, %f)\n", pos.x,
               pos.y, pos.z);
        shift_point_frame_3x4_transform(&pos, ijk_from_world);

        // Calculate the pixel indices for the detector image

        // Convert the hit-location to a vector along the ray from the source
        // to the hit-location such that the vector has unit displacement along
        // the source-to-detector-center line (the vector is not necessarily a
        // unit vector).  Vectors of this sort work with the inverse ray
        // transform.
        pos.x = (pos.x - sx) / sdd;
        pos.y = (pos.y - sy) / sdd;
        pos.z = (pos.z - sz) / sdd;

        shift_vector_frame_3x4_transform(&pos, world_from_ijk);

        /* DEBUG STATEMENT *
        float mag2 = (pos.x * pos.x) + (pos.y * pos.y) + (pos.z * pos.z);
        if (mag2 < 1.f) {
            printf("WARNING: vector to put into inverse ray transform has
        magnitude < 1: %1.10e\n", sqrtf(mag2));
        }
        // should also be project to unit along the detector-plane's normal
        float dotprod = (pos.x * detector_plane->n.x) + (pos.y *
        detector_plane->n.y) + (pos.z * detector_plane->n.z); float normal_norm2
            = (detector_plane->n.x * detector_plane->n.x)
            + (detector_plane->n.y * detector_plane->n.y)
            + (detector_plane->n.z * detector_plane->n.z);
        float proj = fabs(dotprod / sqrtf(normal_norm2));
        if (fabs(1.f - proj) > 1.0e-6f) {
            printf(
                "WARNING: vector to put into inverse ray transform does not has
        unit length " "ALONG source-to-detector direction. Magnitude of
        projection: %1.10e\n", proj
            );
        }
        /**/

        // Use the inverse ray transform. Note that 'pos' is explicitly a
        // homogeneous vector
        int pixel_x =
            (int)((index_from_world[0] * pos.x) +
                  (index_from_world[1] * pos.y) +
                  (index_from_world[2] * pos.z) + (index_from_world[3] * 0.0f));
        int pixel_y =
            (int)((index_from_world[4] * pos.x) +
                  (index_from_world[5] * pos.y) +
                  (index_from_world[6] * pos.z) + (index_from_world[7] * 0.0f));
        //////////printf("pixel: [%d,%d]. num_scatter_events: %d\n", pixel_x,
        ///pixel_y, num_scatter_events);
        if ((pixel_x >= 0) && (pixel_x < detector_width) && (pixel_y >= 0) &&
            (pixel_y < detector_height)) {
          int pixel_index = (pixel_y * detector_width) + pixel_x;
          if (num_scatter_events) {
            // The photon was scattered at least once and thus is not part of
            // the primary
            atomicAdd(&num_scattered_hits[pixel_index], 1);
            // NOTE: atomicAdd(float *, float) only available for compute
            // capability 2.x and higher.
            // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicadd
            atomicAdd(&deposited_energy[pixel_index], energy);
          } else {
            // The photon was not scattered and thus is part of the primary
            atomicAdd(&num_unscattered_hits[pixel_index], 1);
          }
        }
      }
    } else {
      // Check if the photon would travel from the source to hit the detector.
      // Then, if it does, add to num_unscattered_hits since it's part of the
      // X-ray primary

      // frame shifts because detector plane is defined in world coord.s
      shift_point_frame_3x4_transform(&pos, world_from_ijk);
      shift_vector_frame_3x4_transform(&dir, world_from_ijk);
      float dist_to_detector =
          psurface_check_ray_intersection(&pos, &dir, detector_plane);
      shift_point_frame_3x4_transform(&pos, ijk_from_world);
      shift_vector_frame_3x4_transform(&dir, ijk_from_world);

      if (dist_to_detector >= 0.0f) {
        pos.x += dist_to_detector * dir.x;
        pos.y += dist_to_detector * dir.y;
        pos.z += dist_to_detector * dir.z;

        // Convert to ray
        pos.x = (pos.x - sx) / sdd;
        pos.y = (pos.y - sy) / sdd;
        pos.z = (pos.z - sz) / sdd;

        shift_vector_frame_3x4_transform(&pos, world_from_ijk);

        // Use the inverse ray transform. Note that 'pos' is explicitly a
        // homogeneous vector
        int pixel_x =
            (int)((index_from_world[0] * pos.x) +
                  (index_from_world[1] * pos.y) +
                  (index_from_world[2] * pos.z) + (index_from_world[3] * 0.0f));
        int pixel_y =
            (int)((index_from_world[4] * pos.x) +
                  (index_from_world[5] * pos.y) +
                  (index_from_world[6] * pos.z) + (index_from_world[7] * 0.0f));
        // printf("didn't hit volume, but hit detector. pixel: [%d, %d]\n",
        // pixel_x, pixel_y);
        if ((pixel_x >= 0) && (pixel_x < detector_width) && (pixel_y >= 0) &&
            (pixel_y < detector_height)) {
          atomicAdd(&num_unscattered_hits[(pixel_y * detector_width) + pixel_x],
                    1);
        }
      }
    }
  }
  if (histories_printing)
    printf("thread #%d has finished tracking\n", thread_id);

  return;
}

__device__ void track_photon(
    float3_t *pos, // input: initial position in volume. output: end position of
                   // photon history
    float3_t *dir, // input: initial direction
    float *energy, // input: initial energy. output: energy at end of photon
                   // history. Units: [eV]
    int *hits_detector, // Boolean output.  Does the photon actually reach the
                        // detector plane?
    int *num_scatter_events, // should be passed a pointer to an int initialized
                             // to zero.  Returns the number of scatter events
                             // experienced by the photon
    float E_abs, // the energy level below which the photon is assumed to be
                 // absorbed. Units: [eV]
    char *labeled_segmentation,   // [0..NUM_MATERIALS-1]-labeled segmentation
    mat_mfp_data_t *mfp_data_arr, // NUM_MATERIALS-element array of pointers to
                                  // mat_mfp_data_t structs. Material
                                  // associations based on labeled_segmentation
    wc_mfp_data_t *wc_data,
    compton_data_t *compton_arr, // NUM_MATERIALS-element array of pointers to
                                 // compton_data_t.  Material associations as
                                 // with mfp_data_arr
    rayleigh_data_t
        *rayleigh_arr, // NUM_MATERIALS-element array of pointers to rayleigh_t.
                       // Material associations as with mfp_data_arr
    int3_t *volume_shape,          // number of voxels in each direction IJK
    float3_t *gVolumeEdgeMinPoint, // IJK coordinate of minimum bounds of volume
    float3_t *gVolumeEdgeMaxPoint, // IJK coordinate of maximum bounds of volume
    plane_surface_t *detector_plane,
    float *world_from_ijk, // 3x4 transform
    float *ijk_from_world, // 3x4 transform
    rng_seed_t *seed) {
  // NOTE on e_index: it indicates the index of the lower bound of the energy
  // interval the photon is in.
  int e_index; // Update e_index whenever energy is updated
  int vox;     // IJK voxel coord.s of photon, flattened for 1-D array
               // labeled_segmentation
  float mfp_wc, mfp_Ra, mfp_Co, mfp_Tot;
  char curr_mat_id, old_mat_id = -1;

  // Determine initial value of e_index
  e_index = find_energy_index(*energy, mfp_data_arr[0].energy, 0, MAX_MFP_BINS);

  // printf("dir on entry: {%f, %f, %f}\n", dir->x, dir->y, dir->z);
  while (1) {
    vox = get_voxel_1D(pos, gVolumeEdgeMinPoint, gVolumeEdgeMaxPoint,
                       volume_shape);
    // printf("pos: {%f, %f, %f}. vox: %d\n", pos->x, pos->y, pos->z, vox);
    if (vox < 0) {
      break;
    } // photon escaped volume

    get_wc_mfp_data(wc_data, *energy, e_index, &mfp_wc);

    // Delta interactions
    do {
      // simulate moving the photon
      float s = -mfp_wc * logf(ranecu(seed));
      //////////////printf("s: %f -- mfp_wc: %f\n", s, mfp_wc);
      pos->x += s * dir->x;
      pos->y += s * dir->y;
      pos->z += s * dir->z;

      vox = get_voxel_1D(pos, gVolumeEdgeMinPoint, gVolumeEdgeMaxPoint,
                         volume_shape);
      // printf("pos: {%f, %f, %f}. vox: %d\n", pos->x, pos->y, pos->z, vox);
      if (vox < 0) {
        break;
      } // phtoton escaped volume

      curr_mat_id = labeled_segmentation[vox];
      if (curr_mat_id != old_mat_id) {
        // only read the mfp data when necessary
        get_mat_mfp_data(&mfp_data_arr[curr_mat_id], *energy, e_index, &mfp_Ra,
                         &mfp_Co, &mfp_Tot);
        old_mat_id = curr_mat_id;
      }

      // printf(
      //     "mat_id: %d. E: %f, mfp_wc: %f, mfp_Tot:%f, mfp_wc / mfp_Tot:
      //     %f\n", curr_mat_id, *energy, mfp_wc, mfp_Tot, mfp_wc / mfp_Tot
      // );

      // Accept the collision if \xi < mfp_wc / mfp_Tot.
      // Thus, reject the collision if \xi >= mfp_wc / mfp_Tot.
      // See
      // http://serpent.vtt.fi/mediawiki/index.php/Delta-_and_surface-tracking
    } while (ranecu(seed) >= mfp_wc / mfp_Tot);

    /*
     * Here because one of:
     *  1) Photon escaped volume ==> need to break out of interaction loop
     *  2) Accepted the collision ==> already have read the MFP data
     */

    if (vox < 0) {
      break;
    }

    /*
     * Now at a legitimate photon interaction.
     *
     * Sample the photon interaction type
     *
     * (1 / mfp_Tot) * (1 / molecules_per_vol) ==    total interaction cross
     * section =: sigma_Tot (1 / mfp_Ra ) * (1 / molecules_per_vol) == Rayleigh
     * interaction cross section =: sigma_Ra (1 / mfp_Co ) * (1 /
     * molecules_per_vol) ==  Compton interaction cross section =: sigma_Co
     *
     * SAMPLING RULE: Let rnd be a uniformly selected number on [0,1]
     *
     * if rnd < (sigma_Co / sigma_Tot): // if rnd < (mfp_Tot / mfp_Co):
     *   COMPTON INTERACTION
     * elif rnd < ((sigma_Ra + sigma_Co) / sigma_Tot): // if rnd < mfp_Tot * ((1
     * / mfp_Co) + (1 / mfp_Ra)): RAYLEIGH INTERACTION else: OTHER INTERACTION
     * (photoelectric for pair production) ==> photon absorbed
     */
    // printf(
    //     "Scatter event at pos: {%f, %f, %f}\n"
    //     "\tMFPs>>> Ra: [%f], Co: [%f], Tot: [%f]\n"
    //     "\tprob_Co: [%f], (prob_Co + prob_Ra): [%f]\n",
    //     pos->x, pos->y, pos->z,
    //     mfp_Ra, mfp_Co, mfp_Tot,
    //     mfp_Tot / mfp_Co, (mfp_Tot / mfp_Co) + (mfp_Tot / mfp_Ra)
    // );
    double cos_theta;
    float rnd = ranecu(seed);
    float prob_Co = mfp_Tot / mfp_Co;
    if (rnd < prob_Co) {
      cos_theta = sample_Compton(energy, &compton_arr[curr_mat_id], seed);
      e_index =
          find_energy_index(*energy, mfp_data_arr[0].energy, 0, e_index + 1);
    } else if (rnd < (prob_Co + (mfp_Tot / mfp_Ra))) {
      cos_theta =
          sample_Rayleigh(*energy, e_index, &rayleigh_arr[curr_mat_id], seed);
    } else {
      *hits_detector = 0;
      return;
    }

    if (*energy < E_abs) {
      *hits_detector = 0;
      return;
    }

    (*num_scatter_events)++;

    double phi = TWO_PI_DOUBLE * ranecu_double(seed);
    get_scattered_dir(dir, cos_theta, phi, world_from_ijk, ijk_from_world);
    // printf("dir has changed to: {%f, %f, %f}\n", dir->x, dir->y, dir->z);
  }

  /* Final processing once the photon has left the volume */

  // Transport the photon to the detector plane
  // frame shifts because detector plane is defined in world coord.s
  shift_point_frame_3x4_transform(pos, world_from_ijk);
  shift_vector_frame_3x4_transform(dir, world_from_ijk);
  float dist_to_detector =
      psurface_check_ray_intersection(pos, dir, detector_plane);
  shift_point_frame_3x4_transform(pos, ijk_from_world);
  shift_vector_frame_3x4_transform(dir, ijk_from_world);
  if (dist_to_detector < 0.0f) {
    *hits_detector = 0;
  } else {
    pos->x += dist_to_detector * dir->x;
    pos->y += dist_to_detector * dir->y;
    pos->z += dist_to_detector * dir->z;
    *hits_detector = 1;
    // NOTE: the calculation for determine which pixel is done in caller
    // function
  }
}

__device__ int get_voxel_1D(float3_t *pos, float3_t *gVolumeEdgeMinPoint,
                            float3_t *gVolumeEdgeMaxPoint,
                            int3_t *volume_shape) {
  /*
   * Returns index into a flattened 1-D array that represents the volume.
   * If outside volume, returns a negative value.
   *
   * volume_arr_3D[x, y, z] == volume_arr_1D[z * y_len * z_len + y * z_len + z]
   *
   * That way, it's like indexing into a 3D array:
   *
   * DTYPE volume_arr[x_len][y_len][z_len];
   */
  if ((pos->x < gVolumeEdgeMinPoint->x /*+ VOXEL_EPS*/) ||
      (pos->x > gVolumeEdgeMaxPoint->x /*- VOXEL_EPS*/) ||
      (pos->y < gVolumeEdgeMinPoint->y /*+ VOXEL_EPS*/) ||
      (pos->y > gVolumeEdgeMaxPoint->y /*- VOXEL_EPS*/) ||
      (pos->z < gVolumeEdgeMinPoint->z /*+ VOXEL_EPS*/) ||
      (pos->z > gVolumeEdgeMaxPoint->z /*- VOXEL_EPS*/)) {
    // Photon outside volume
    return -1;
  }
  int vox_x, vox_y, vox_z;
  vox_x = (int)(pos->x - gVolumeEdgeMinPoint->x);
  vox_y = (int)(pos->y - gVolumeEdgeMinPoint->y);
  vox_z = (int)(pos->z - gVolumeEdgeMinPoint->z);

  return (vox_x * volume_shape->y * volume_shape->z) +
         (vox_y * volume_shape->z) + vox_z;
}

__device__ void
get_scattered_dir(float3_t *dir, // direction: both input and output. IJK space
                  double cos_theta,      // polar scattering angle
                  double phi,            // azimuthal scattering angle
                  float *world_from_ijk, // 3x4 transformation matrix
                  float *ijk_from_world  // 3x4 transformation matrix
) {
  // TODO: once scatter is working, this can be sped up by keeping track of both
  // "dir_world" and "dir_ijk"
  //  in the main track_photon(...) loop, which would eliminate about of the mat
  //  mult frame conversion
  shift_vector_frame_3x4_transform(dir, world_from_ijk);

  // Since \theta is restricted to [0,\pi], sin_theta is restricted to [0,1]
  float cos_th = (float)cos_theta;
  float sin_th = (float)sqrt(1.0 - cos_theta * cos_theta);
  float cos_phi = (float)cos(phi);
  float sin_phi = (float)sin(phi);

  if ((cos_th < -1.f) || (cos_th > 1.f)) {
    printf("cos_th outside valid range of [-1,1]. cos_th=%f\n", cos_th);
  }

  float tmp = sqrtf(1.f - dir->z * dir->z);

  if (tmp < 1.0e-8f) {
    tmp = 1.0e-8f; // to avoid divide-by-zero errors
  }

  float orig_x = dir->x;

  dir->x = dir->x * cos_th +
           sin_th * (dir->x * dir->z * cos_phi - dir->y * sin_phi) / tmp;
  dir->y = dir->y * cos_th +
           sin_th * (dir->y * dir->z * cos_phi - orig_x * sin_phi) / tmp;
  dir->z = dir->z * cos_th - sin_th * tmp * cos_phi;

  float mag = (dir->x * dir->x) + (dir->y * dir->y) +
              (dir->z * dir->z); // actually magnitude^2

  if (fabs(mag - 1.0f) > 1.0e-14) {
    // Only do the computationally expensive normalization when necessary
    mag = sqrtf(mag);

    dir->x /= mag;
    dir->y /= mag;
    dir->z /= mag;
  }

  // convert back to IJK
  shift_vector_frame_3x4_transform(dir, ijk_from_world);
}

__device__ void move_photon_to_volume(
    float3_t *pos,    // position of the photon.  Serves as both input and ouput
    float3_t *dir,    // input: direction of photon travel
    int *hits_volume, // Boolean output.  Does the photon actually hit the
                      // volume?
    float3_t *gVolumeEdgeMinPoint, // IJK coordinate of minimum bounds of volume
    float3_t *gVolumeEdgeMaxPoint  // IJK coordinate of maximum bounds of volume
) {
  /*
   * Strategy: calculate the which direction out of {x,y,z} needs to travel the
   * most to get to the volume.  This determines how far the photon must travel
   * if it has any hope of reaching the volume. Next, will need to do checks to
   * ensure that the resulting position is inside of the volume.
   */
  float dist_x, dist_y, dist_z;
  /* Calculations for x-direction */
  if (dir->x > VOXEL_EPS) {
    if (pos->x > gVolumeEdgeMinPoint->x) {
      // Photon inside or past volume
      dist_x = 0.0f;
    } else {
      // Add VOXEL_EPS to make super sure that the photon reaches the volume
      dist_x = VOXEL_EPS + (gVolumeEdgeMinPoint->x - pos->x) / dir->x;
    }
  } else if (dir->x < NEG_VOXEL_EPS) {
    if (pos->x < gVolumeEdgeMaxPoint->x) {
      dist_x = 0.0f;
    } else {
      // In order to ensure that dist_x is positive, we divide the negative
      // quantity (gVolumeEdgeMaxPoint->x - pos->x) by the negative quantity
      // 'dir->x'.
      dist_x = VOXEL_EPS + (gVolumeEdgeMaxPoint->x - pos->x) / dir->x;
    }
  } else {
    // No collision with an x-normal-plane possible
    dist_x = NEG_INFTY;
  }

  /* Calculations for y-direction */
  if (dir->y > VOXEL_EPS) {
    if (pos->y > gVolumeEdgeMinPoint->y) {
      // Photon inside or past volume
      dist_y = 0.0f;
    } else {
      // Add VOXEL_EPS to make super sure that the photon reaches the volume
      dist_y = VOXEL_EPS + (gVolumeEdgeMinPoint->y - pos->y) / dir->y;
    }
  } else if (dir->y < NEG_VOXEL_EPS) {
    if (pos->y < gVolumeEdgeMaxPoint->y) {
      dist_y = 0.0f;
    } else {
      // In order to ensure that dist_y is positive, we divide the negative
      // quantity (gVolumeEdgeMaxPoint->y - pos->y) by the negative quantity
      // 'dir->y'.
      dist_y = VOXEL_EPS + (gVolumeEdgeMaxPoint->y - pos->y) / dir->y;
    }
  } else {
    // No collision with an y-normal-plane possible
    dist_y = NEG_INFTY;
  }

  /* Calculations for z-direction */
  if (dir->z > VOXEL_EPS) {
    if (pos->z > gVolumeEdgeMinPoint->z) {
      // Photon inside or past volume
      dist_z = 0.0f;
    } else {
      // Add VOXEL_EPS to make super sure that the photon reaches the volume
      dist_z = VOXEL_EPS + (gVolumeEdgeMinPoint->z - pos->z) / dir->z;
    }
  } else if (dir->z < NEG_VOXEL_EPS) {
    if (pos->z < gVolumeEdgeMaxPoint->z) {
      dist_z = 0.0f;
    } else {
      // In order to ensure that dist_z is positive, we divide the negative
      // quantity (gVolumeEdgeMaxPoint->z - pos->z) by the negative quantity
      // 'dir->z'.
      dist_z = VOXEL_EPS + (gVolumeEdgeMaxPoint->z - pos->z) / dir->z;
    }
  } else {
    // No collision with an y-normal-plane possible
    dist_z = NEG_INFTY;
  }

  /*
   * Store the longest distance to a plane in dist_z.
   * If distance if zero: interpret as photon already in volume, or no
   * intersection is possible (for example, if the photon is moving away)
   */
  dist_z = MAX_VAL(dist_z, MAX_VAL(dist_x, dist_y));

  // Move the photon to the volume (yay! the whole purpose of the function!)
  pos->x += dist_z * dir->x;
  pos->y += dist_z * dir->y;
  pos->z += dist_z * dir->z;

  /*
   * Final error checking. Check if the new position is outside the volume.
   * If so, move the particle back to original position and set the intersection
   * flag to false.
   */
  if ((pos->x < gVolumeEdgeMinPoint->x) || (pos->x > gVolumeEdgeMaxPoint->x) ||
      (pos->y < gVolumeEdgeMinPoint->y) || (pos->y > gVolumeEdgeMaxPoint->y) ||
      (pos->z < gVolumeEdgeMinPoint->z) || (pos->z > gVolumeEdgeMaxPoint->z)) {
    pos->x -= dist_z * dir->x;
    pos->y -= dist_z * dir->y;
    pos->z -= dist_z * dir->z;
    *hits_volume = 0;
  } else {
    *hits_volume = 1;
  }
}

__device__ void sample_initial_dir_world(
    float3_t *dir, // output: the direction in world coordinates, sampled
                   // uniformly from the unit sphere
    rng_seed_t *seed) {
  // Sampling explanation here:
  // http://corysimon.github.io/articles/uniformdistn-on-sphere/
  double phi = TWO_PI_DOUBLE * ranecu_double(seed);
  double theta = acos(1.0 - 2.0 * ranecu_double(seed));

  double sin_theta = sin(theta);

  dir->x = (float)(sin_theta * cos(phi));
  dir->y = (float)(sin_theta * sin(phi));
  dir->z = (float)(cos(theta));
}

__device__ void shift_point_frame_3x4_transform(
    float3_t *pt, // [in/out]: the point to be transformed
    float *transform) {
  float x = pt->x;
  float y = pt->y;
  float z = pt->z;

  pt->x = (transform[0] * x) + (transform[1] * y) + (transform[2] * z) +
          (transform[3] * 1.0f);
  pt->y = (transform[4] * x) + (transform[5] * y) + (transform[6] * z) +
          (transform[7] * 1.f);
  pt->z = (transform[8] * x) + (transform[9] * y) + (transform[10] * z) +
          (transform[11] * 1.0f);
}

__device__ void shift_vector_frame_3x4_transform(
    float3_t *vec, // [in/out]: the vector to be transformed
    float *transform) {
  float x = vec->x;
  float y = vec->y;
  float z = vec->z;

  vec->x = (transform[0] * x) + (transform[1] * y) + (transform[2] * z) +
           (transform[3] * 0.0f);
  vec->y = (transform[4] * x) + (transform[5] * y) + (transform[6] * z) +
           (transform[7] * 0.0f);
  vec->z = (transform[8] * x) + (transform[9] * y) + (transform[10] * z) +
           (transform[11] * 0.0f);
}

__device__ float sample_initial_energy(const int n_bins,
                                       const float *spectrum_energies,
                                       const float *spectrum_cdf,
                                       rng_seed_t *seed) {
  float threshold = ranecu(seed);

  // Binary search to find the interval [CDF(i), CDF(i+1)] that contains
  // 'threshold'
  int lo_idx = 0;      // inclusive
  int hi_idx = n_bins; // exclusive
  int i;
  while (lo_idx < hi_idx) {
    i = (lo_idx + hi_idx) / 2;

    // Check if 'i' is the lower bound of the correct interval
    if (threshold < spectrum_cdf[i]) {
      // Need to check lower intervals
      hi_idx = i;
    } else if (threshold < spectrum_cdf[i + 1]) {
      // Found the correct interval
      break;
    } else {
      // Need to check higher intervals
      lo_idx = i + 1;
    }
  }

  /* DEBUG STATEMENT *
  if (spectrum_cdf[i] > threshold) {
      printf(
          "ERROR: sample_initial_energy identified too-high interval.
  threshold=%.10f, spectrum_cdf[i]=%.10f\n", threshold, spectrum_cdf[i]
      );
  } else if (spectrum_cdf[i+1] <= threshold) {
      printf(
          "ERROR: sample_initial_energy identified too-low interval.
  threshold=%.10f, spectrum_cdf[i+1]=%.10f\n", threshold, spectrum_cdf[i+1]
      );
  }
  /**/

  // Final interpolation within the spectral bin
  float slope = (spectrum_energies[i + 1] - spectrum_energies[i]) /
                (spectrum_cdf[i + 1] - spectrum_cdf[i]);

  return spectrum_energies[i] + (slope * (threshold - spectrum_cdf[i]));
}

__device__ double sample_rita(const rayleigh_data_t *sampler,
                              const double pmax_current, rng_seed_t *seed) {
  double y = ranecu_double(seed) * pmax_current;

  // Binary search to find the interval [y_i, y_{i+1}] that contains y
  int lo_idx = 0;                  // inclusive
  int hi_idx = sampler->n_gridpts; // exclusive
  int i;
  while (lo_idx < hi_idx) {
    i = (lo_idx + hi_idx) / 2;

    // Check if 'i' is the lower bound of the correct interval
    if (y < sampler->y[i]) {
      // Need to check lower intervals
      hi_idx = i;
    } else if (y < sampler->y[i + 1]) {
      // Found correct interval
      break;
    } else {
      // Need to check higher intervals
      lo_idx = i + 1;
    }
  }

  /* DEBUG STATEMENT *
  if (sampler->y[i] > y) {
      printf("ERROR: RITA identified too-high interval. y=%.10f,
  y[i=%d]=%.10f\n", y, i, sampler->y[i]); } else if (sampler->y[i+1] <= y) {
      printf("ERROR: RITA identified too-low interval. y=%.10f,
  y[i+1=%d]=%.10f\n", y, i+1, sampler->y[i+1]);
  }
  /**/

  double nu = y - sampler->y[i];
  if (nu > 1e-16) { // this logic takes great 'inspiration' from MCGPU
    double delta_i = sampler->y[i + 1] - sampler->y[i];

    // Avoid multiple accesses to the same global variable
    float a_i = sampler->a[i];
    float b_i = sampler->b[i];
    float x_i = sampler->x[i];

    double tmp = (delta_i * delta_i) +
                 ((a_i * delta_i) + (b_i * nu)) * nu; // denominator
    tmp = (1.0 + a_i + b_i) * delta_i * nu / tmp;     // numerator / denominator
    return (double)x_i + (tmp * (double)(sampler->x[i + 1] - x_i));
  } else {
    return sampler->x[i];
  }
}

__device__ float psurface_check_ray_intersection(
    float3_t *pos,              // input: current position of the photon in IJK
    float3_t *dir,              // input: direction of photon travel in IJK
    const plane_surface_t *psur // detector plane struct, in world coord.s
) {
  /*
   * If there will be an intersection, returns the distance to the intersection.
   * If no intersection, returns a negative number (the negative number does not
   * necessarily have a geometrical meaning)
   *
   * Let \vec{m} be the 'plane vector'.
   * (\vec{pos} + \alpha * \vec{dir}) \cdot \vec{m} = 0,
   * then (\vec{pos} + \alpha * \vec{dir}) is the point of intersection.
   */
  float r_dot_m = (pos->x * psur->n.x) + (pos->y * psur->n.y) +
                  (pos->z * psur->n.z) + psur->d;
  if (0.0f == r_dot_m) {
    // Photon is already on the plane
    return 0.0f;
  }
  float d_dot_m =
      (dir->x * psur->n.x) + (dir->y * psur->n.y) + (dir->z * psur->n.z);
  if (0.0f == d_dot_m) {
    // Direction of photon travel is perpendicular to the normal vector of the
    // plane Thus, there will be no intersection
    return -1.f;
  }
  return -1.f * r_dot_m / d_dot_m;
}

__device__ int find_energy_index(float nrg, float *energy_arr,
                                 int lo_idx, // inclusive
                                 int hi_idx  // exclusive
) {
  // Binary search to find the interval [E_i, E_{i+1} that contains nrg]
  // NOTE: param energy_arr is generally a field of mat_mfp_data_t, etc
  if ((nrg < energy_arr[lo_idx]) || (nrg >= energy_arr[hi_idx])) {
    return -1;
  }

  int i;
  while (lo_idx < hi_idx) {
    i = (lo_idx + hi_idx) / 2;

    // Check if 'i' is the lower bound of the correct interval
    if (nrg < energy_arr[i]) {
      // Need to check lower intervals
      hi_idx = i;
    } else if (nrg < energy_arr[i + 1]) {
      // Found the correct interval
      break;
    } else {
      // Need to check higher intervals
      lo_idx = i + 1;
    }
  }
  return i;
}

__device__ void get_mat_mfp_data(
    mat_mfp_data_t *data,
    float nrg,   // energy of the photon
    int e_index, // the index of the lower bound of the energy interval
    float *ra,   // output: MFP for Rayleigh scatter. Units: [mm]
    float *co,   // output: MFP for Compton scatter. Units: [mm]
    float *tot   // output: MFP (total). Units: [mm]
) {
  /* DEBUG STATEMENT *
  if (nrg < data->energy[0]) {
      printf(
          "ERROR: photon energy (%6f) less than minimum "
          "material MFP data energy level (%6f)\n", nrg, data->energy[0]
      );
  } else if (nrg > data->energy[data->n_bins - 1]) {
      printf(
          "ERROR: photon energy (%6f) greater than maximum "
          "material MFP data energy level (%6f)\n",
          nrg, data->energy[data->n_bins - 1]
      );
  }
  if (e_index < 0) {
      printf(
          "ERROR: e_index (%d) less than minimum (0), energy level = %6f\n",
          e_index, nrg
      );
  } else if (e_index >= data->n_bins - 1) {
      printf(
          "ERROR: e_index (%d) more than maximum (%d), energy level = %6f\n",
          e_index, data->n_bins - 2, nrg
      );
  }
  /**/

  // printf("i=%d. energy[i]=%f, nrg=%f, energy[i+1]=%f\n", i, data->energy[i],
  // nrg, data->energy[i+1]);

  // linear interpolation
  float alpha = (nrg - data->energy[e_index]) /
                (data->energy[e_index + 1] - data->energy[e_index]);
  float one_minus_alpha = 1.f - alpha;

  *ra = (one_minus_alpha * data->mfp_Ra[e_index]) +
        (alpha * data->mfp_Ra[e_index + 1]);
  *co = (one_minus_alpha * data->mfp_Co[e_index]) +
        (alpha * data->mfp_Co[e_index + 1]);
  *tot = (one_minus_alpha * data->mfp_Tot[e_index]) +
         (alpha * data->mfp_Tot[e_index + 1]);

  // printf(
  //     "alpha: %f, i: %d. mfp_Tot[i]=%f, mfp_Tot[i+1]=%f"
  //     "Ra: %f, Co: %f, Tot: %f\n",
  //     alpha, i, data->mfp_Tot[i], data->mfp_Tot[i+1],
  //     *ra, *co, *tot
  // );

  return;
}

__device__ void get_wc_mfp_data(
    wc_mfp_data_t *data,
    float nrg,   // energy of the photon [eV]
    int e_index, // the index of the lower bound of the energy interval
    float *mfp   // output: Woodcock MFP. Units: [mm]
) {
  /* DEBUG STATEMENT *
  if (nrg < data->energy[0]) {
      printf(
          "ERROR: photon energy (%6f) less than minimum "
          "Woodcock MFP data energy level (%6f)\n", nrg, data->energy[0]
      );
  } else if (nrg > data->energy[data->n_bins - 1]) {
      printf(
          "ERROR: photon energy (%6f) greater than maximum "
          "Woodcock MFP data energy level (%6f)\n",
          nrg, data->energy[data->n_bins - 1]
      );
  }
  if (e_index < 0) {
      printf(
          "ERROR: e_index (%d) less than minimum (0), energy level = %6f\n",
          e_index, nrg
      );
  } else if (e_index >= data->n_bins - 1) {
      printf(
          "ERROR: e_index (%d) more than maximum (%d), energy level = %6f\n",
          e_index, data->n_bins - 2, nrg
      );
  }
  /**/

  /* DEBUG STATEMENT *
  if (data->energy[i] > nrg) {
      printf(
          "ERROR: Woodcock MFP data identified too-high energy bin. "
          "nrg=%6e, data->energy[i]=%6e\n", nrg, data->energy[i]
      );
  } else if (data->energy[i+1] <= nrg) {
      printf(
          "ERROR: Woodcock MFP data identified too-low energy bin. "
          "nrg=%6e, data->energy[i+1]=%6e\n", nrg, data->energy[i+1]
      );
  }
  /**/

  // linear interpolation
  float alpha = (nrg - data->energy[e_index]) /
                (data->energy[e_index + 1] - data->energy[e_index]);

  *mfp = ((1.f - alpha) * data->mfp_wc[e_index]) +
         (alpha * data->mfp_wc[e_index + 1]);

  return;
}

__device__ double sample_Rayleigh(
    float energy,
    int e_index, // the index of the lower bound of the energy interval
    const rayleigh_data_t *rayleigh_data, rng_seed_t *seed) {
  double pmax_current = (double)rayleigh_data->pmax[e_index + 1];

  // Sample a random value of x^2 from the distribution pi(x^2), restricted to
  // the interval (0, x_{max}^2)
  double x_max2;
  // double kappa = ((double)energy) * (double)INV_ELECTRON_REST_ENERGY;
  // x_max2 = 424.66493476 * 4.0 * kappa * kappa;
  x_max2 =
      (energy * energy) * 6.50528656295084103e-9; // the constant is (2.0
                                                  // * 20.6074 / 510998.918) ^ 2
  x_max2 =
      MIN_VAL(x_max2, (double)rayleigh_data->x[rayleigh_data->n_gridpts - 1]);

  double cos_theta;
  if (x_max2 < 0.0001) {
    do {
      cos_theta = 1.0 - ranecu_double(seed) * 2.0;
    } while (ranecu_double(seed) > 0.5 * (1.0 + (cos_theta * cos_theta)));
    return cos_theta;
  }

  float x2;
  while (1) { // Loop will iterate every time the sampled value is rejected or
              // above maximum
    x2 = sample_rita(rayleigh_data, pmax_current, seed);

    if (x2 < x_max2) {
      // Set cos_theta
      cos_theta = 1.0 - (2.0 * x2 / x_max2);

      // Test cos_theta
      // double g = (1.0 + cos_theta * cos_theta) * 0.5;
      // Reject and re-sample if \xi > g; accept if \xi < g
      if (ranecu_double(seed) < ((1.0 + cos_theta * cos_theta) * 0.5)) {
        return cos_theta;
      }
    }
  }
}

__device__ double
sample_Compton(float *energy, // serves as both input and output
               const compton_data_t *compton_data, rng_seed_t *seed) {
  float kappa = (*energy) * INV_ELECTRON_REST_ENERGY;
  float one_p2k = 1.f + 2.f * kappa;
  float tau_min = 1.f / one_p2k;

  float a_1 = logf(one_p2k);
  float a_2 = 2.f * kappa * (1.f * kappa) / (one_p2k * one_p2k);

  /* Sample cos_theta */

  // Compute S(E, \theta=\pi) here, since it does not depend on cos_theta
  float s_pi = 0.f;
  for (int shell = 0; shell < compton_data->nshells; shell++) {
    float tmp = compton_data->ui[shell];
    if (*energy > tmp) { // this serves as the Heaviside function
      float left_term =
          (*energy) * (*energy - tmp) * 2.f; // since (1 - \cos(\theta)) == 2
      float piomc = (left_term - ELECTRON_REST_ENERGY * tmp) /
                    (ELECTRON_REST_ENERGY *
                     sqrtf(left_term + left_term +
                           tmp * tmp)); // PENELOPE p_{i,max} / (m_{e} c)

      tmp = compton_data->jmc[shell] *
            piomc; // this now contains the PENELOPE value: J_{i,0} * p_{i,max}
      if (piomc < 0) {
        tmp = (1.f - tmp - tmp);
      } else {
        tmp = (1.f + tmp + tmp);
      }
      tmp = 0.5f - (0.5f * tmp * tmp); // calculating exponent
      tmp = 0.5 * expf(tmp);
      if (piomc > 0) {
        tmp = 1.f - tmp;
      }
      // 'tmp' now holds PENELOPE n_{i}(p_{i,max})

      s_pi += (compton_data->f[shell] *
               tmp); // Equivalent to: s_pi += f_{i} n_{i}(p_{i,max})
    }
  }

  // local storage for the results of calculating n_{i}(p_{i,max})
  float n_pimax_vals[MAX_NSHELLS];
  float tau;
  double one_minus_cos;
  float T_tau_term, s_theta;

  do {
    /* Sample tau */
    if (ranecu(seed) < (a_1 / (a_1 + a_2))) {
      // i == 1
      tau = powf(tau_min, ranecu(seed));
    } else {
      // i == 2
      tau = sqrtf(1.f + (tau_min * tau_min - 1.f) * ranecu(seed));
      /*
       * Explanation: PENELOPE uses the term \tau_{min}^2 + \xi * (1 -
       * \tau_{min}^2)
       *  == 1 - (1 - \tau_{min}^2) + \xi * (1 - \tau_{min}^2)
       *  == 1 + [(1 - \tau_{min}^2) * (-1 + \xi)]
       *  == 1 + [(\tau_{min}^2 - 1) * (1 - \xi)]
       *  == 1 + (\tau_{min}^2 - 1) * \xi,
       * since \xi is uniformly distributed on the interval [0,1].
       */
    }
    one_minus_cos = (1.0 - (double)tau) / ((double)kappa * (double)tau);

    s_theta = 0.0f;
    for (int shell = 0; shell < compton_data->nshells; shell++) {
      float tmp = compton_data->ui[shell];
      if (*energy > tmp) { // this serves as the Heaviside function
        float left_term = (*energy) * (*energy - tmp) * ((float)one_minus_cos);
        float piomc = (left_term - ELECTRON_REST_ENERGY * tmp) /
                      (ELECTRON_REST_ENERGY *
                       sqrtf(left_term + left_term +
                             tmp * tmp)); // PENELOPE p_{i,max} / (m_{e} c)

        tmp =
            compton_data->jmc[shell] *
            piomc; // this now contains the PENELOPE value: J_{i,0} * p_{i,max}
        if (piomc < 0) {
          tmp = (1.f - tmp - tmp);
        } else {
          tmp = (1.f + tmp + tmp);
        }
        tmp = 0.5f - (0.5f * tmp * tmp); // calculating exponent
        tmp = 0.5 * expf(tmp);
        if (piomc > 0) {
          tmp = 1.f - tmp;
        }
        // 'tmp' now holds PENELOPE n_{i}(p_{i,max})

        s_theta += (compton_data->f[shell] *
                    tmp); // Equivalent to: s_pi += f_{i} n_{i}(p_{i,max})
        n_pimax_vals[shell] = tmp;
      }
    }

    // Compute the term of T(cos_theta) that does not involve S(E, \theta)
    T_tau_term = kappa * kappa * tau * (1.f + tau * tau); // the denominator
    T_tau_term = (T_tau_term - (1.f - tau) * (one_p2k * tau - 1.f)) /
                 T_tau_term; // the whole expression

    // Reject and re-sample if \xi > T(cos_theta)
    // Thus, reject and re-sample if (\xi * S(\theta=\pi)) > (T_tau_term *
    // S(\theta))
  } while ((ranecu(seed) * s_pi) > (T_tau_term * s_theta));

  // cos_theta is set by now
  float cos_theta = 1.f - one_minus_cos;

  /* Choose the active shell */
  float pzomc; // "P_Z Over M_{e} C" == p_z / (m_{e} c)
  float F_p_z, F_max;

  do {
    /*
     * Steps:
     *  1. Choose a threshold value in range [0, s_theta]
     *  2. Accumulate the partial sum of f_{i} \Theta(E - U_i) n_{i}(p_{i,max})
     * over the electron shells
     *  3. Once the partial sum reaches the threshold value, we 'return' the
     * most recently considered shell. In this manner, we select the active
     * electron shell with relative probability equal to f_{i} \Theta(E - U_i)
     * n_{i}(p_{i,max}).
     *  4. Calculate a random value of p_z
     *  5. Reject p_z and start over if p_z < -1 * m_{e} * c
     *  6. Calculate F_{max} and F_{p_z} and reject appropriately
     */
    float threshold = ranecu(seed) * s_theta;
    float accumulator = 0.0f;
    int shell;
    for (shell = 0; shell < compton_data->nshells - 1; shell++) {
      /*
       * End condition makes it such that if the first (nshells-1) shells don't
       * reach threshold, the loop will automatically set active_shell to the
       * last shell number
       */
      accumulator += compton_data->f[shell] * n_pimax_vals[shell];
      if (accumulator >= threshold) {
        break;
      }
    }

    float two_A = ranecu(seed) * (2.f * n_pimax_vals[shell]);
    if (two_A < 1) {
      pzomc = 0.5f - sqrtf(0.25f - 0.5f * logf(two_A));
    } else {
      pzomc = sqrtf(0.25f - 0.5f * logf(2.f - two_A)) - 0.5f;
    }
    pzomc = pzomc / compton_data->jmc[shell];

    if (pzomc < -1.f) {
      // Erroneous (physically impossible) value obtained due to numerical
      // errors. Re-sample
      continue;
    }

    // Calculate F(p_z) from PENELOPE-2006
    float tmp = 1.f + (tau * tau) -
                (2.f * tau *
                 cos_theta); // tmp = (\beta)^2, where \beta := (c q_{C}) / E
    tmp = sqrtf(tmp) * (1.f + tau * (tau - cos_theta) / tmp);
    F_p_z = 1.f + (tmp * pzomc);
    F_max = 1.f + (tmp * 0.2f);
    if (pzomc < 0) {
      F_max = -1.f * F_max;
    }
    // TODO: refactor the above calculation so the comparison btwn F_max and
    // F_p_z does not use division operations

    // Accept if (\xi * F_max) < F_p_z
    // Thus, reject and re-sample if (\xi * F_max) >= F_p_z
  } while ((ranecu(seed) * F_max) >= F_p_z);

  // pzomc is now set. Calculate E_ration = E_prime / E
  float t = pzomc * pzomc;
  float term_tau = 1.f - t * tau * tau;
  float term_cos = 1.f - t * tau * ((float)cos_theta);

  float E_ratio = sqrtf(term_cos * term_cos - term_tau * (1.f - t));
  if (pzomc < 0) {
    E_ratio = -1.f * E_ratio;
  }
  E_ratio = tau * (term_cos + E_ratio) / term_tau;

  *energy = E_ratio * (*energy);
  return cos_theta;
}

__device__ inline float ranecu(rng_seed_t *seed) {
  // Implementation from PENELOPE-2006 section 1.2
  int i = (int)(seed->x / 53668); // "i1"
  seed->x = 40014 * (seed->x - (i * 53668)) - (i * 12211);
  if (seed->x < 0) {
    seed->x = seed->x + 2147483563;
  }

  // no longer need "i1", so variable 'i' refers to "i2"
  i = (int)(seed->y / 52774);
  seed->y = 40692 * (seed->y - (i * 52774)) - (i * 3791);
  if (seed->y < 0) {
    seed->y = seed->y + 2147483399;
  }

  // no longer need "i2", so variable 'i' refers to "iz"
  i = seed->x - seed->y;
  if (i < 1) {
    i = i + 2147483562;
  }

  // double uscale = 1.0 / (2.147483563e9);
  // uscale is approx. equal to 4.65661305739e-10
  return ((float)i) * 4.65661305739e-10f;
}

__device__ inline double ranecu_double(rng_seed_t *seed) {
  // basically the same as ranecu(...), but converting to double at the end
  // Implementation from PENELOPE-2006 section 1.2
  int i = (int)(seed->x / 53668); // "i1"
  seed->x = 40014 * (seed->x - (i * 53668)) - (i * 12211);
  if (seed->x < 0) {
    seed->x = seed->x + 2147483563;
  }

  // no longer need "i1", so variable 'i' refers to "i2"
  i = (int)(seed->y / 52774);
  seed->y = 40692 * (seed->y - (i * 52774)) - (i * 3791);
  if (seed->y < 0) {
    seed->y = seed->y + 2147483399;
  }

  // no longer need "i2", so variable 'i' refers to "iz"
  i = seed->x - seed->y;
  if (i < 1) {
    i = i + 2147483562;
  }

  // double uscale = 1.0 / (2.147483563e9);
  // uscale is approx. equal to 4.6566130573917692e-10
  return ((double)i) * 4.6566130573917692e-10;
}

/* Maximum number of random values sampled per photon */
#define LEAP_DISTANCE 256
/* RANECU values */
#define a1_RANECU 40014
#define m1_RANECU 2147483563
#define a2_RANECU 40692
#define m2_RANECU 2147483399
__device__ void initialize_seed(
    int thread_id, // each CUDA thread should have a unique ID given to it
    int histories_for_thread, int seed_input, rng_seed_t *seed) {
  /* From MC-GPU */
  // Initialize first MLCG
  unsigned long long int leap = ((unsigned long long int)(thread_id + 1)) *
                                (histories_for_thread * LEAP_DISTANCE);
  int y = 1;
  int z = a1_RANECU;
  // Use modular artihmetic to compute (a^leap)MOD(m)
  while (1) {
    if (0 != (leap & 01)) {
      // leap is odd
      leap >>= 1;                  // leap = leap/2
      y = abMODm(m1_RANECU, z, y); // y = (z * y) MOD m
      if (0 == leap) {
        break;
      }
    } else {
      // leap is even
      leap >>= 1; // leap = leap/2
    }
    z = abMODm(m1_RANECU, z, z); // z = (z * z) MOD m
  }
  // Here, y = (a^j) MOD m

  // seed(i+j) = [(a^j MOD m) * seed(i)] MOD m
  seed->x = abMODm(m1_RANECU, seed_input, y);

  // Initialize second MLCG
  leap = ((unsigned long long int)(thread_id + 1)) *
         (histories_for_thread * LEAP_DISTANCE);
  y = 1;
  z = a2_RANECU;
  // Use modular artihmetic to compute (a^leap)MOD(m)
  while (1) {
    if (0 != (leap & 01)) {
      // leap is odd
      leap >>= 1;                  // leap = leap/2
      y = abMODm(m2_RANECU, z, y); // y = (z * y) MOD m
      if (0 == leap) {
        break;
      }
    } else {
      // leap is even
      leap >>= 1; // leap = leap/2
    }
    z = abMODm(m2_RANECU, z, z); // z = (z * z) MOD m
  }
  // Here, y = (a^j) MOD m
  seed->y = abMODm(m2_RANECU, seed_input, y);

  return;
}

__device__ inline int abMODm(int m, int a1, int a2) {
  // COMPUTE (a1 * a2) MOD m
  int q, k;
  int p = -m; // negative to avoid overflow when adding

  // Apply Russian peasant method until "a <= 32768"
  while (a1 > 32768) { // 32-bit ints: 2^(('32'-2)/2) = 32768
    if (0 != (a1 & 1)) {
      // Store a2 when a1 is odd
      p += a2;
      if (p > 0) {
        p -= m;
      }
    }
    a1 >>= 1;           // a1 = a1 / 2
    a2 = (a2 - m) + a2; // double a2 (MOD m)
    if (a2 < 0) {
      a2 += m; // ensure a2 is always positive
    }
  }

  // Approximate factoring method, since a1 is small enough to avoid overflow
  q = (int)m / a1;
  k = (int)a2 / q;
  a2 = a1 * (a2 - (k * q)) - k * (m - (q * a1));
  while (a2 < 0) {
    a2 += m;
  }

  // Final processing
  p += a2;
  if (p < 0) {
    p += m;
  }
  return p;
}
}
