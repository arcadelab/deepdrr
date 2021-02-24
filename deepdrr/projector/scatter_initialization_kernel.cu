/*
 * Based on Sisniega et al. (2015), "High-fidelity artifact correction for cone-beam CT imaging of the brain"
 */

#include "scatter_initialization_header.cu"

extern "C" {
    __global__ void initialization_stage(
        int detector_width, // size of detector in pixels 
        int detector_height,
        int histories_for_thread, // number of photons for -this- thread to track
        char *nominal_segmentation, // [0..2]-labeled segmentation obtained by thresholding: [-infty, -500, 300, infty]
        float sx, // coordinates of source in IJK
        float sy, // (not in a float3_t for ease of calling from Python wrapper)
        float sz,
        int volume_shape_x, // integer size of the volume to avoid 
        int volume_shape_y, // floating-point errors with the gVolumeEdge{Min,Max}Point
        int volume_shape_z, // and gVoxelElementSize math
        float gVolumeEdgeMinPointX, // bounds of the volume in IJK
        float gVolumeEdgeMinPointY,
        float gVolumeEdgeMinPointZ,
        float gVolumeEdgeMaxPointX,
        float gVolumeEdgeMaxPointY,
        float gVolumeEdgeMaxPointZ,
        float gVoxelElementSizeX, // voxel size in IJK
        float gVoxelElementSizeY,
        float gVoxelElementSizeZ,
        float *index_from_ijk, // (2, 4) array giving the IJK-homogeneous-coord.s-to-pixel-coord.s transformation
        mat_mfp_data_t *air_mfp,
        mat_mfp_data_t *soft_mfp,
        mat_mfp_data_t *bone_mfp,
        wc_mfp_data_t *woodcock_mfp,
        compton_data_t *air_Co_data,
        compton_data_t *soft_Co_data,
        compton_data_t *bone_Co_data,
        rita_t *air_rita,
        rita_t *soft_rita,
        rita_t *bone_rita,
        plane_surface_t *detector_plane,
        int n_bins, // the number of spectral bins
        float *spectrum_energies, // 1-D array -- size is the n_bins
        float *spectrum_cdf, // 1-D array -- cumulative density function over the energies
        float E_abs, // the energy level below which photons are assumed to be absorbed
        float *deposited_energy // the output.  Size is [detector_width]x[detector_height]
    ) {
        rng_seed_t seed; // TODO: initialize

        int3_t volume_shape = {
            .x = volume_shape_x,
            .y = volume_shape_y,
            .z = volume_shape_z
        };
        float3_t gVolumeEdgeMinPoint = {
            .x = gVolumeEdgeMinPointX,
            .y = gVolumeEdgeMinPointY,
            .z = gVolumeEdgeMinPointZ
        };
        float3_t gVolumeEdgeMaxPoint = {
            .x = gVolumeEdgeMaxPointX,
            .y = gVolumeEdgeMaxPointY,
            .z = gVolumeEdgeMaxPointZ
        };
        float3_t gVoxelElementSize = {
            .x = gVoxelElementSizeX,
            .y = gVoxelElementSizeY,
            .z = gVoxelElementSizeZ
        };

        mat_mfp_data_t *mfp_data_arr[3];
        mfp_data_arr[NOM_SEG_AIR_ID] = air_mfp;
        mfp_data_arr[NOM_SEG_SOFT_ID] = soft_mfp;
        mfp_data_arr[NOM_SEG_BONE_ID] = bone_mfp;

        compton_data_t *compton_arr[3];
        compton_arr[NOM_SEG_AIR_ID] = air_Co_data;
        compton_arr[NOM_SEG_SOFT_ID] = soft_Co_data;
        compton_arr[NOM_SEG_BONE_ID] = bone_Co_data;

        rita_t *rita_arr[3];
        rita_arr[NOM_SEG_AIR_ID] = air_rita;
        rita_arr[NOM_SEG_SOFT_ID] = soft_rita;
        rita_arr[NOM_SEG_BONE_ID] = bone_rita;

        for (; histories_for_thread > 0; histories_for_thread--) {
            float energy = sample_initial_energy(n_bins, spectrum_energies, spectrum_cdf, seed);
            float3_t pos = { .x = sx, .y = sy, .z = sz };
            float3_t dir;
            sample_initial_dir(&dir, &seed);
            int is_hit;
            move_photon_to_volume(&pos, &dir, &is_hit, &gVolumeEdgeMinPoint, &gVolumeEdgeMaxPoint);
            if (is_hit) {
                // is_hit gets repurposed since we don't need it anymore for 'did the photon hit the volume'
                int num_scatter_events = 0;
                initialization_track_photon(
                    &pos, &dir, &energy, &is_hit, &num_scatter_events,
                    E_abs, nominal_segmentation, 
                    mfp_data_arr, woodcock_mfp, compton_arr, rita_arr,
                    &volume_shape, 
                    &gVolumeEdgeMinPoint, &gVolumeEdgeMaxPoint, &gVoxelElementSize,
                    detector_plane, &seed
                );

                if (is_hit && num_scatter_events) {
                    // The photon was scattered at least once and thus is not part of the primary
                    // 'pos' contains the IJK coord.s of collision with the detector.
                    // Calculate the pixel indices for the detector image
                    int pixel_x = (int)((index_from_ijk[0] * pos->x) + (index_from_ijk[1] * pos->y) + (index_from_ijk[2] * pos->z) + index_from_ijk[3]);
                    int pixel_y = (int)((index_from_ijk[4] * pos->x) + (index_from_ijk[5] * pos->y) + (index_from_ijk[6] * pos->z) + index_from_ijk[7]);
                    if ((pixel_x >= 0) && (pixel_x < detector_width) && (pixel_y >= 0) && (pixel_y < detector_height)) {
                        // NOTE: atomicAdd(float *, float) only available for compute capability 2.x and higher.
                        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicadd
                        atomicAdd(&deposited_energy[(pixel_y * detector_width) + pixel_x], energy);
                    }
                }
            } else {
                /*
                 * Do not need to check if the photon hits the detector here since
                 * that photon would be considered part of the primary X-ray image
                 */
            }
        }

        return;
    }

    __device__ void initialization_track_photon(
        float3_t *pos, // input: initial position in volume. output: end position of photon history
        float3_t *dir, // input: initial direction
        float *energy, // input: initial energy. output: energy at end of photon history
        int *hits_detector, // Boolean output.  Does the photon actually reach the detector plane?
        int *num_scatter_events, // should be passed a pointer to an int initialized to zero.  Returns the number of scatter events experienced by the photon
        float E_abs, // the energy level below which the photon is assumed to be absorbed
        char *labeled_segmentation, // [0..2]-labeled segmentation obtained by thresholding: [-infty, -500, 300, infty]
        mat_mfp_data_t **mfp_data_arr, // 3-element array of pointers to mat_mfp_data_t structs. Idx NOM_SEG_AIR_ID associated with air, etc
        wc_mfp_data_t *wc_data,
        compton_data_t **compton_arr, // 3-element array of pointers to compton_data_t.  Material associations as with mfp_data_arr
        rita_t **rita_arr, // 3-element array of pointers to rita_t.  Material associations as with mfp_data_arr
        int3_t *volume_shape, // number of voxels in each direction IJK
        float3_t *gVolumeEdgeMinPoint, // IJK coordinate of minimum bounds of volume
        float3_t *gVolumeEdgeMaxPoint, // IJK coordinate of maximum bounds of volume
        float3_t *gVoxelElementSize, // IJK coordinate lengths of each dimension of a voxel
        plane_surface_t *detector_plane, 
        rng_seed_t *seed
    ) {
        int vox; // IJK voxel coord.s of photon, flattened for 1-D array labeled_segmentation
        float mfp_wc, mfp_Ra, mfp_Co, mfp_Tot;
        while (1) {
            vox = get_voxel_1D(pos, gVolumeEdgeMinPoint, gVolumeEdgeMaxPoint, gVoxelElementSize, volume_shape);
            if (vox < 0) { break; } // photon escaped volume

            mfp_wc = get_wc_mfp_data(wc_data, *energy, &mfp_wc);

            // Delta interactions
            do {
                // simulate moving the photon
                float s = -mfp_wc * logf(ranecu(seed));
                pos->x += s * dir->x;
                pos->y += s * dir->y;
                pos->z += s * dir->z;

                vox = get_voxel_1D(pos, gVolumeEdgeMinPoint, gVolumeEdgeMaxPoint, gVoxelElementSize, volume_shape);
                if (vox < 0) { break; } // phtoton escaped volume

                char mat_id = labeled_segmentation[vox]
                get_mat_mfp_data(mfp_data_arr[mat_id], *energy, &mfp_Ra, &mfp_Co, &mfp_Tot);

                // Accept the collision if \xi < mfp_wc / mfp_Tot.
                // Thus, reject the collision if \xi >= mfp_wc / mfp_Tot.
                // See http://serpent.vtt.fi/mediawiki/index.php/Delta-_and_surface-tracking
            } while (ranecu(seed) >= mfp_wc / mfp_Tot);

            char mat_id = labeled_segmentation[vox]
            get_mat_mfp_data(mfp_data_arr[mat_id], *energy, &mfp_Ra, &mfp_Co, &mfp_Tot);
            
            /* 
             * Now at a legitimate photon interaction. 
             * 
             * Sample the photon interaction type
             *
             * (1 / mfp_Tot) * (1 / molecules_per_vol) ==    total interaction cross section =: sigma_Tot
             * (1 / mfp_Ra ) * (1 / molecules_per_vol) == Rayleigh interaction cross section =: sigma_Ra
             * (1 / mfp_Co ) * (1 / molecules_per_vol) ==  Compton interaction cross section =: sigma_Co
             *
             * SAMPLING RULE: Let rnd be a uniformly selected number on [0,1]
             *
             * if rnd < (sigma_Co / sigma_Tot): // if rnd < (mfp_Tot / mfp_Co):
             *   COMPTON INTERACTION
             * elif rnd < ((sigma_Ra + sigma_Co) / sigma_Tot): // if rnd < mfp_Tot * ((1 / mfp_Co) + (1 / mfp_Ra)):
             *   RAYLEIGH INTERACTION
             * else:
             *   OTHER INTERACTION (photoelectric for pair production) ==> photon absorbed
             */
            double cos_theta;
            float rnd = ranecu(seed);
            float prob_Co = mfp_Tot / mfp_Co;
            if (rnd < prob_Co) {
                cos_theta = sample_Compton(energy, compton_arr[mat_id], seed);
            } else if (rnd < (prob_Co + (mfp_Tot / mfp_Ra))) {
                cos_theta = sample_Rayleigh(*energy, rita_arr[mat_id], seed);
            } else {
                *hits_detector = 0;
                return;
            }

            if (*energy < E_abs) {
                *hits_detector = 0;
                return;
            }

            (*num_scatter_events)++;

            phi = TWO_PI_DOUBLE * ranecu_double(seed);
            get_scattered_dir(dir, cos_theta, phi);
        }

        /* Final processing once the photon has left the volume */

        // Transport the photon to the detector plane
        float dist_to_detector = psurface_check_ray_intersection(pos, dir, detector_plane);
        if (dist_to_detector < 0.0f) {
            *hits_detector = 0;
        }

        pos->x += dist_to_detector * dir->x;
        pos->y += dist_to_detector * dir->y;
        pos->z += dist_to_detector * dir->z;
        *hits_detector = 1;
        // NOTE: the calculation for determine which pixel is done in caller function
    }

    __device__ int get_voxel_1D(
        float3_t *pos,
        float3_t *gVolumeEdgeMinPoint,
        float3_t *gVolumeEdgeMaxPoint,
        float3_t *gVoxelElementSize,
        int3_t *volume_shape
    ) {
        /* 
         * Returns index into a flattened 1-D array that represents the volume.  
         * If outside volume, returns a negative value.
         *
         * volume_arr_3D[x, y, z] == volume_arr_1D[z * x_len * y_len + y * x_len + x]
         */
        if ((pos->x < gVolumeEdgeMinPoint->x + VOXEL_EPS) || (pos->x > gVolumeEdgeMaxPoint->x - VOXEL_EPS) ||
                (pos->y < gVolumeEdgeMinPoint->y + VOXEL_EPS) || (pos->y > gVolumeEdgeMaxPoint->y - VOXEL_EPS) ||
                (pos->z < gVolumeEdgeMinPoint->z + VOXEL_EPS) || (pos->z > gVolumeEdgeMaxPoint->z - VOXEL_EPS) ) {
            // Photon outside volume
            return -1;
        }
        int vox_x, vox_y, vox_z;
        vox_x = (int)((pos->x - gVolumeEdgeMinPoint->x) / gVoxelElementSize->x);
        vox_y = (int)((pos->y - gVolumeEdgeMinPoint->y) / gVoxelElementSize->y);
        vox_z = (int)((pos->z - gVolumeEdgeMinPoint->z) / gVoxelElementSize->z);

        return (vox_z * volume_shape->x * volume_shape->y) + (vox_y * volume_shape->x) + vox_x;
    }

    __device__ void get_scattered_dir(
        float3_t *dir, // direction: both input and output
        double cos_theta, // polar scattering angle
        double phi // azimuthal scattering angle
    ) {
        // Since \theta is restricted to [0,\pi], sin_theta is restricted to [0,1]
        float cos_th  = (float)cos_theta;
        float sin_th  = (float)sqrt(1.0 - cos_theta * cos_theta);
        float cos_phi = (float)cos(phi);
        float sin_phi = (float)sin(phi);

        float tmp = sqrtf(1.f - dir->z * dir->z);

        float orig_x = dir->x;

        dir->x = dir->x * cos_th + sin_th * (dir->x * dir->z * cos_phi - dir->y * sin_phi) / tmp;
        dir->y = dir->y * cos_th + sin_th * (dir->y * dir->z * cos_phi - orig_x * sin_phi) / tmp;
        dir->z = dir->z * cos_th - sin_th * tmp * cos_phi;

        float mag = (dir->x * dir->x) + (dir->y * dir->y) + (dir->z * dir->z); // actually magnitude^2

        if (fabs(mag - 1.0f) > 1.0e-14) {
            // Only do the computationally expensive normalization when necessary
            mag = sqrtf(mag);

            dir->x /= mag;
            dir->y /= mag;
            dir->z /= mag;
        }
    }

    __device__ void move_photon_to_volume(
        float3_t *pos, // position of the photon.  Serves as both input and ouput
        float3_t *dir, // input: direction of photon travel
        int *hits_volume, // Boolean output.  Does the photon actually hit the volume?
        float3_t *gVolumeEdgeMinPoint, // IJK coordinate of minimum bounds of volume
        float3_t *gVolumeEdgeMaxPoint, // IJK coordinate of maximum bounds of volume
    ) {
        /*
         * Strategy: calculate the which direction out of {x,y,z} needs to travel the most to get
         * to the volume.  This determines how far the photon must travel if it has any hope of 
         * reaching the volume.
         * Next, will need to do checks to ensure that the resulting position is inside of the volume.
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
                // quantity (gVolumeEdgeMaxPoint->x - pos->x) by the negative quantity 'dir->x'.
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
                // quantity (gVolumeEdgeMaxPoint->y - pos->y) by the negative quantity 'dir->y'.
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
                // quantity (gVolumeEdgeMaxPoint->z - pos->z) by the negative quantity 'dir->z'.
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

        // Move the photon to the volume (yay! the whole purpose of this function!)
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
                (pos->z < gVolumeEdgeMinPoint->z) || (pos->z > gVolumeEdgeMaxPoint->z) ) {
            pos->x -= dist_z * dir->x;
            pos->y -= dist_z * dir->y;
            pos->z -= dist_z * dir->z;
            *hits_volume = 0;
        } else {
            *hits_volume = 1;
        }
    }

    __device__ void sample_initial_dir(
        float3_t *dir, // output: the sampled direction
        rng_seed_t *seed
    ) {
        // Sampling explanation here: http://corysimon.github.io/articles/uniformdistn-on-sphere/
        double phi = TWO_PI_DOUBLE * ranecu_double(seed);
        double theta = acos(1.0 - 2.0 * ranecu_double(seed));

        double sin_theta = sin(theta);
        
        dir->x = (float)(sin_theta * cos(phi));
        dir->y = (float)(sin_theta * sin(phi));
        dir->z = (float)(cos(theta));
    }

    __device__ float sample_initial_energy(
        const int n_bins,
        const float *spectrum_energies,
        const float *spectrum_cdf,
        rng_seed_t *seed
    ) {
        float threshold = ranecu(seed);

        // Binary search to find the interval [CDF(i), CDF(i+1)] that contains 'threshold'
        int lo_idx = 0; // inclusive
        int hi_idx = n_bins; // exclusive
        int i;
        while (lo_idx < hi_idx) {
            i = (lo_idx + hi_idx) / 2; 

            // Check if 'i' is the lower bound of the correct interval
            if (threshold < spectrum_cdf[i]) {
                // Need to check lower intervals
                hi_idx = i;
            } else if (threshold < spectrum_cdf[i+1]) {
                // Found the correct interval
                break;
            } else {
                // Need to check higher intervals
                lo_idx = i + 1;
            }
        }

        /* DEBUG STATEMENT
        if (spectrum_cdf[i] > threshold) {
            printf(
                "ERROR: sample_initial_energy identified too-high interval. threshold=%.10f, spectrum_cdf[i]=%.10f\n", 
                threshold, spectrum_cdf[i]
            );
        }
        if (spectrum_cdf[i+1] <= threshold) {
            printf(
                "ERROR: sample_initial_energy identified too-low interval. threshold=%.10f, spectrum_cdf[i+1]=%.10f\n", 
                threshold, spectrum_cdf[i+1]
            );
        }
        */

        // Final interpolation within the spectral bin
        float slope = (spectrum_energies[i+1] - spectrum_energies[i]) / (spectrum_cdf[i+1] - spectrum_cdf[i])

        return spectrum_energies[i] + (slope * (threshold - spectrum_cdf[i]));
    }

    __device__ double sample_rita(
        const rita_t *sampler,
        rng_seed_t *seed
    ) {
        double y = ranecu_double(seed);

        // Binary search to find the interval [y_i, y_{i+1}] that contains y
        int lo_idx = 0; // inclusive
        int hi_idx = sampler->n_gridpts; // exclusive
        int i;
        while (lo_idx < hi_idx) {
            i = (lo_idx + hi_idx) / 2;

            // Check if 'i' is the lower bound of the correct interval
            if (y < sampler->y[i]) {
                // Need to check lower intervals
                hi_idx = i;
            } else if (y < sampler->y[i+1]) {
                // Found correct interval
                break;
            } else {
                // Need to check higher intervals
                lo_idx = i + 1;
            }
        }

        /* DEBUG STATEMENT
        if (sampler->y[i] > y) {
            printf("ERROR: RITA identified too-high interval. y=%.10f, y[i]=%.10f\n", y, sampler->y[i]);
        }
        if (sampler->y[i+1] <= y) {
            printf("ERROR: RITA identified too-low interval. y=%.10f, y[i+1]=%.10f\n", y, sampler->y[i+1]);
        }
        */

        double nu = y - sampler->y[i];
        double delta_i = sampler->y[i+1] - sampler->y[i];

        double tmp = (delta_i * delta_i) + (sampler->a[i] * delta_i * nu) + (sampler->b[i] * nu * nu); // denominator
        tmp = (1.0 + sampler->a[i] + sampler->b[i]) * delta_i * nu / tmp; // numerator / denominator

        return sampler->x[i] + (tmp * (sampler->x[i+1] - sampler->x[i]));
    }

    __device__ float psurface_check_ray_intersection(
        float3_t *pos, // input: current position of the photon
        float3_t *dir, // input: direction of photon travel
        const plane_surface_t *psur
    ) {
        /*
         * If there will be an intersection, returns the distance to the intersection.
         * If no intersection, returns a negative number (the negative number does not necessarily have a
         * geometrical meaning) 
         *
         * Let \vec{m} be the 'plane vector'.
         * (\vec{pos} + \alpha * \vec{dir}) \cdot \vec{m} = 0, 
         * then (\vec{pos} + \alpha * \vec{dir}) is the point of intersection.
         */
        float r_dot_m = (pos->x * psur->n.x) + (pos->y * psur->n.y) + (pos->z * psur->n.z) + psur->d;
        if (0.0f == r_dot_m) {
            // Photon is already on the plane
            return 0.0f;
        }
        float d_dot_m = (dir->x * psur->n.x) + (dir->y * psur->n.y) + (dir->z * psur->n.z);
        if (0.0f == d_dot_m) {
            // Direction of photon travel is perpendicular to the normal vector of the plane
            // Thus, there will be no intersection
            return -1.f;
        }
        return -1.f * r_dot_m / d_dot_m;
    }

    __device__ void get_mat_mfp_data(
        mat_mfp_data_t *data,
        float nrg, // energy of the photon
        float *ra, // output: MFP for Rayleigh scatter. Units: [mm]
        float *co, // output: MFP for Compton scatter. Units: [mm]
        float *tot // output: MFP (total). Units: [mm]
    ) {
        // TODO: implement (using binary search)
        return;
    }

    __device__ void get_wc_mfp_data(
        wc_mfp_data_t *data,
        float nrg, // energy of the photon [eV]
        float *mfp // output: Woodcock MFP. Units: [mm]
    ) {
        // TODO: implement (using binary search)
        return;
    }

    __device__ double sample_Rayleigh(
        float energy,
        const rita_t *ff_sampler,
        rng_seed_t *seed
    ) {
        double kappa = ((double)energy) * (double)INV_ELECTRON_REST_ENERGY;
        // Sample a random value of x^2 from the distribution pi(x^2), restricted to the interval (0, x_{max}^2)
        double x_max2 = 424.66493476 * 4.0 * kappa * kappa;
        float x2;
        do {
            x2 = sample_rita(ff_sampler, seed);
        } while (x2 > x_max2);

        double cos_theta;
        do {
            // Set cos_theta
            cos_theta = 1.0 - (2.0 * x2 / x_max2);

            // Test cos_theta
            //double g = (1.0 + cos_theta * cos_theta) * 0.5;
            
            // Reject and re-sample if \xi > g
        } while (ranecu_double(seed) > ((1.0 + cos_theta * cos_theta) * 0.5));

        return cos_theta;
    }

    __device__ double sample_Compton(
        float *energy, // serves as both input and output
        const compton_data_t *compton_data,
        rng_seed_t *seed
    ) {
        float kappa = *energy * INV_ELECTRON_REST_ENERGY;
        float one_p2k = 1.f + 2.f * kappa;
        float tau_min = 1.f / one_p2k;

        float a_1 = logf(one_p2k);
        float a_1 = 2.f * kappa * (1.f * kappa) / (one_p2k * one_p2k);

        /* Sample cos_theta */

        // Compute S(E, \theta=\pi) here, since it does not depend on cos_theta
        float s_pi = 0.f;
        for (int shell = 0; shell < compton_data->nshells; shell++) {
            float tmp = compton_data->ui[shell];
            if (*energy > tmp) { // this serves as the Heaviside function
                float left_term = (*energy) * (*energy - tmp) * 2.f; // since (1 - \cos(\theta)) == 2
                float piomc = (left_term - ELECTRON_REST_ENERGY * tmp) / (ELECTRON_REST_ENERGY * sqrtf(left_term + left_term + tmp * tmp)); // PENELOPE p_{i,max} / (m_{e} c)

                tmp = compton_data->jmc[shell] * piomc; // this now contains the PENELOPE value: J_{i,0} * p_{i,max}
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

                s_pi += (compton_data->f[shell] * tmp); // Equivalent to: s_pi += f_{i} n_{i}(p_{i,max})
            }
        }

        double cos_theta;
        // local storage for the results of calculating n_{i}(p_{i,max})
        float n_pimax_vals[MAX_NSHELLS];
        float tau;
        double one_minus_cos;

        do {
            /* Sample tau */
            if (ranecu(seed) < (a1 / (a1 + a2))) {
                // i == 1
                tau = powf(tau_min, ranecu(seed));
            } else {
                // i == 2
                tau = sqrtf(1.f + (tau_min * tau_min - 1.f) * ranecu(seed));
                /*
                 * Explanation: PENELOPE uses the term \tau_{min}^2 + \xi * (1 - \tau_{min}^2)
                 *  == 1 - (1 - \tau_{min}^2) + \xi * (1 - \tau_{min}^2)
                 *  == 1 + [(1 - \tau_{min}^2) * (-1 + \xi)]
                 *  == 1 + [(\tau_{min}^2 - 1) * (1 - \xi)]
                 *  == 1 + (\tau_{min}^2 - 1) * \xi,
                 * since \xi is uniformly distributed on the interval [0,1].
                 */
            }
            one_minus_cos = (1.0 - (double)tau) / ((double)kappa * (double)tau);

            float s_theta = 0.0f;
            for (int shell = 0; shell < compton_data->nshells; shell++) {
                float tmp = compton_data->ui[shell];
                if (*energy > tmp) { // this serves as the Heaviside function
                    float left_term = (*energy) * (*energy - tmp) * ((float)one_minus_cos);
                    float piomc = (left_term - ELECTRON_REST_ENERGY * tmp) / (ELECTRON_REST_ENERGY * sqrtf(left_term + left_term + tmp * tmp)); // PENELOPE p_{i,max} / (m_{e} c)

                    tmp = compton_data->jmc[shell] * piomc; // this now contains the PENELOPE value: J_{i,0} * p_{i,max}
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

                    s_pi += (compton_data->f[shell] * tmp); // Equivalent to: s_pi += f_{i} n_{i}(p_{i,max})
                    n_pimax_vals[shell] = tmp;
                }
            }
            
            // Compute the term of T(cos_theta) that does not involve S(E, \theta)
            float T_tau_term = kappa * kappa * tau * (1.f + tau * tau); // the denominator
            T_tau_term = (T_tau_term - (1.f - tau) * (one_p2k * tau - 1.f)) / T_tau_term; // the whole expression

            // Reject and re-sample if \xi > T(cos_theta)
            // Thus, reject and re-sample if (\xi * S(\theta=\pi)) > (T_tau_term * S(\theta))
        } while ((ranecu(seed) * s_pi) > (T_tau_term * s_theta));
        
        // cos_theta is set by now
        float cos_theta = 1.f - one_minus_cos;

        /* Choose the active shell */
        float pzomc; // "P_Z Over M_{e} C" == p_z / (m_{e} c)

        do {
            /*
             * Steps:
             *  1. Choose a threshold value in range [0, s_theta]
             *  2. Accumulate the partial sum of f_{i} \Theta(E - U_i) n_{i}(p_{i,max}) over the electron shells
             *  3. Once the partial sum reaches the threshold value, we 'return' the most recently considered 
             *      shell. In this manner, we select the active electron shell with relative probability equal 
             *      to f_{i} \Theta(E - U_i) n_{i}(p_{i,max}).
             *  4. Calculate a random value of p_z
             *  5. Reject p_z and start over if p_z < -1 * m_{e} * c
             *  6. Calculate F_{max} and F_{p_z} and reject appropriately
             */
            float threshold = ranecu(seed) * s_theta;
            float accumulator = 0.0f;
            int shell;
            for (shell = 0; shell < compton_data->nshells - 1; shell++) {
                /*
                 * End condition makes it such that if the first (nshells-1) shells don't reach threshold,
                 * the loop will automatically set active_shell to the last shell number
                 */
                accumulator += compton_data->f[shell] * n_pimax_vals[shell];
                if (accumulator >= threshold) {
                    break;
                }
            }

            two_A = ranecu(seed) * (2.f * n_pimax_vals[shell]);
            if (two_A < 1) {
                pzomc = 0.5f - sqrtf(0.25f - 0.5f * logf(two_A));
            } else {
                pzomc = sqrtf(0.25f - 0.5f * logf(2.f - two_A)) - 0.5f;
            }
            pzomc = pzomc / compton_data->jmc[shell];

            if (pzomc < -1.f) {
                // Erroneous (physically impossible) value obtained due to numerical errors. Re-sample
                continue;
            }

            // Calculate F(p_z) from PENELOPE-2006
            float tmp = 1.f + (tau * tau) - (2.f * tau * cos_theta); // tmp = (\beta)^2, where \beta := (c q_{C}) / E
            tmp = sqrtf(tmp) * (1.f + tau * (tau - cos_theta) / tmp);
            float F_p_z = 1.f + (tmp * pzomc);
            float F_max = 1.f + (tmp * 0.2f);
            if (pzomc < 0) {
                F_max = -1.f * F_max;
            }
            // TODO: refactor the above calculation so the comparison btwn F_max and F_p_z does not use division operations

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

    inline float ranecu(rng_seed_t *seed) {
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

    inline double ranecu_double(rng_seed_t *seed) {
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
}