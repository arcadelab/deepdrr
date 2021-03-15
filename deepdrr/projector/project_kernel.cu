#include <stdio.h>
#include <cubicTex3D.cu>

#include "project_kernel_data.cu"

#define UPDATE(multiplier, vol_id, mat_id) do {\
    adiatl[(mat_id)] = (multiplier) * tex3D(VOLUME(vol_id), px, py, pz) * round(cubicTex3D(SEG(vol_id, mat_id), px, py, pz)) * volume_normalization_factor[vol_id];\
} while (0)

#define GET_POSITION_FOR_VOL(vol_id) do {\
    /* Get the current sample point in the volume voxel-space. */\
    /* In CUDA, voxel centeras are located at (xx.5, xx.5, xx.5), whereas SwVolume has voxel centers at integers. */\
    px = sx[vol_id] + alpha * rx[vol_id] - gVolumeEdgeMinPointX[vol_id];\
    py = sy[vol_id] + alpha * ry[vol_id] - gVolumeEdgeMinPointY[vol_id];\
    pz = sz[vol_id] + alpha * rz[vol_id] - gVolumeEdgeMinPointZ[vol_id];\
} while (0)

#if NUM_MATERIALS == 1
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    GET_POSITION_FOR_VOL(vol_id);\
    UPDATE(multiplier, vol_id, 0);\
} while (0)
#elif NUM_MATERIALS == 2
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    GET_POSITION_FOR_VOL(vol_id);\
    UPDATE(multiplier, vol_id, 0);\
    UPDATE(multiplier, vol_id, 1);\
} while (0)
#elif NUM_MATERIALS == 3
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    GET_POSITION_FOR_VOL(vol_id);\
    UPDATE(multiplier, vol_id, 0);\
    UPDATE(multiplier, vol_id, 1);\
    UPDATE(multiplier, vol_id, 2);\
} while (0)
#elif NUM_MATERIALS == 4
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    GET_POSITION_FOR_VOL(vol_id);\
    UPDATE(multiplier, vol_id, 0);\
    UPDATE(multiplier, vol_id, 1);\
    UPDATE(multiplier, vol_id, 2);\
    UPDATE(multiplier, vol_id, 3);\
} while (0)
#elif NUM_MATERIALS == 5
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    GET_POSITION_FOR_VOL(vol_id);\
    UPDATE(multiplier, vol_id, 0);\
    UPDATE(multiplier, vol_id, 1);\
    UPDATE(multiplier, vol_id, 2);\
    UPDATE(multiplier, vol_id, 3);\
    UPDATE(multiplier, vol_id, 4);\
} while (0)
#elif NUM_MATERIALS == 6
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    GET_POSITION_FOR_VOL(vol_id);\
    UPDATE(multiplier, vol_id, 0);\
    UPDATE(multiplier, vol_id, 1);\
    UPDATE(multiplier, vol_id, 2);\
    UPDATE(multiplier, vol_id, 3);\
    UPDATE(multiplier, vol_id, 4);\
    UPDATE(multiplier, vol_id, 5);\
} while (0)
#elif NUM_MATERIALS == 7
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    GET_POSITION_FOR_VOL(vol_id);\
    UPDATE(multiplier, vol_id, 0);\
    UPDATE(multiplier, vol_id, 1);\
    UPDATE(multiplier, vol_id, 2);\
    UPDATE(multiplier, vol_id, 3);\
    UPDATE(multiplier, vol_id, 4);\
    UPDATE(multiplier, vol_id, 5);\
    UPDATE(multiplier, vol_id, 6);\
} while (0)
#elif NUM_MATERIALS == 8
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    GET_POSITION_FOR_VOL(vol_id);\
    UPDATE(multiplier, vol_id, 0);\
    UPDATE(multiplier, vol_id, 1);\
    UPDATE(multiplier, vol_id, 2);\
    UPDATE(multiplier, vol_id, 3);\
    UPDATE(multiplier, vol_id, 4);\
    UPDATE(multiplier, vol_id, 5);\
    UPDATE(multiplier, vol_id, 6);\
    UPDATE(multiplier, vol_id, 7);\
} while (0)
#elif NUM_MATERIALS == 9
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    GET_POSITION_FOR_VOL(vol_id);\
    UPDATE(multiplier, vol_id, 0);\
    UPDATE(multiplier, vol_id, 1);\
    UPDATE(multiplier, vol_id, 2);\
    UPDATE(multiplier, vol_id, 3);\
    UPDATE(multiplier, vol_id, 4);\
    UPDATE(multiplier, vol_id, 5);\
    UPDATE(multiplier, vol_id, 6);\
    UPDATE(multiplier, vol_id, 7);\
    UPDATE(multiplier, vol_id, 8);\
} while (0)
#elif NUM_MATERIALS == 10
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    GET_POSITION_FOR_VOL(vol_id);\
    UPDATE(multiplier, vol_id, 0);\
    UPDATE(multiplier, vol_id, 1);\
    UPDATE(multiplier, vol_id, 2);\
    UPDATE(multiplier, vol_id, 3);\
    UPDATE(multiplier, vol_id, 4);\
    UPDATE(multiplier, vol_id, 5);\
    UPDATE(multiplier, vol_id, 6);\
    UPDATE(multiplier, vol_id, 7);\
    UPDATE(multiplier, vol_id, 8);\
    UPDATE(multiplier, vol_id, 9);\
} while (0)
#elif NUM_MATERIALS == 11
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    GET_POSITION_FOR_VOL(vol_id);\
    UPDATE(multiplier, vol_id, 0);\
    UPDATE(multiplier, vol_id, 1);\
    UPDATE(multiplier, vol_id, 2);\
    UPDATE(multiplier, vol_id, 3);\
    UPDATE(multiplier, vol_id, 4);\
    UPDATE(multiplier, vol_id, 5);\
    UPDATE(multiplier, vol_id, 6);\
    UPDATE(multiplier, vol_id, 7);\
    UPDATE(multiplier, vol_id, 8);\
    UPDATE(multiplier, vol_id, 9);\
    UPDATE(multiplier, vol_id, 10);\
} while (0)
#elif NUM_MATERIALS == 12
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    GET_POSITION_FOR_VOL(vol_id);\
    UPDATE(multiplier, vol_id, 0);\
    UPDATE(multiplier, vol_id, 1);\
    UPDATE(multiplier, vol_id, 2);\
    UPDATE(multiplier, vol_id, 3);\
    UPDATE(multiplier, vol_id, 4);\
    UPDATE(multiplier, vol_id, 5);\
    UPDATE(multiplier, vol_id, 6);\
    UPDATE(multiplier, vol_id, 7);\
    UPDATE(multiplier, vol_id, 8);\
    UPDATE(multiplier, vol_id, 9);\
    UPDATE(multiplier, vol_id, 10);\
    UPDATE(multiplier, vol_id, 11);\
} while (0)
#elif NUM_MATERIALS == 13
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    GET_POSITION_FOR_VOL(vol_id);\
    UPDATE(multiplier, vol_id, 0);\
    UPDATE(multiplier, vol_id, 1);\
    UPDATE(multiplier, vol_id, 2);\
    UPDATE(multiplier, vol_id, 3);\
    UPDATE(multiplier, vol_id, 4);\
    UPDATE(multiplier, vol_id, 5);\
    UPDATE(multiplier, vol_id, 6);\
    UPDATE(multiplier, vol_id, 7);\
    UPDATE(multiplier, vol_id, 8);\
    UPDATE(multiplier, vol_id, 9);\
    UPDATE(multiplier, vol_id, 10);\
    UPDATE(multiplier, vol_id, 11);\
    UPDATE(multiplier, vol_id, 12);\
} while (0)
#elif NUM_MATERIALS == 14
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    GET_POSITION_FOR_VOL(vol_id);\
    UPDATE(multiplier, vol_id, 0);\
    UPDATE(multiplier, vol_id, 1);\
    UPDATE(multiplier, vol_id, 2);\
    UPDATE(multiplier, vol_id, 3);\
    UPDATE(multiplier, vol_id, 4);\
    UPDATE(multiplier, vol_id, 5);\
    UPDATE(multiplier, vol_id, 6);\
    UPDATE(multiplier, vol_id, 7);\
    UPDATE(multiplier, vol_id, 8);\
    UPDATE(multiplier, vol_id, 9);\
    UPDATE(multiplier, vol_id, 10);\
    UPDATE(multiplier, vol_id, 11);\
    UPDATE(multiplier, vol_id, 12);\
    UPDATE(multiplier, vol_id, 13);\
} while (0)
#else
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    fprintf(stderr, "NUM_MATERIALS not in [1, 14]");\
} while (0)
#endif

#if NUM_VOLUMES == 1
#define INTERPOLATE(multiplier) do {\
    if (priority[0] == curr_priority) { INTERPOLATE_FOR_VOL(multiplier, 0); }\
} while (0)
#elif NUM_VOLUMES == 2
#define INTERPOLATE(multiplier) do {\
    if (priority[0] == curr_priority) { INTERPOLATE_FOR_VOL(multiplier, 0); }\
    if (priority[1] == curr_priority) { INTERPOLATE_FOR_VOL(multiplier, 1); }\
} while (0)
#elif NUM_VOLUMES == 3
#define INTERPOLATE(multiplier) do {\
    if (priority[0] == curr_priority) { INTERPOLATE_FOR_VOL(multiplier, 0); }\
    if (priority[1] == curr_priority) { INTERPOLATE_FOR_VOL(multiplier, 1); }\
    if (priority[2] == curr_priority) { INTERPOLATE_FOR_VOL(multiplier, 2); }\
} while (0)
#else
#define INTERPOLATE(multiplier) do {\
    fprintf(stderr, "INTERPOLATE not supported for NUM_VOLUMES outside [1, 3]");\
} while (0)
#endif

#define CALCULATE_RAY_FOR_VOL(vol_id) do {\
    rx[vol_id] = u * rt_kinv[(9 * vol_id) + 0] + v * rt_kinv[(9 * vol_id) + 1] + rt_kinv[(9 * vol_id) + 2];\
    ry[vol_id] = u * rt_kinv[(9 * vol_id) + 3] + v * rt_kinv[(9 * vol_id) + 4] + rt_kinv[(9 * vol_id) + 5];\
    rz[vol_id] = u * rt_kinv[(9 * vol_id) + 6] + v * rt_kinv[(9 * vol_id) + 7] + rt_kinv[(9 * vol_id) + 8];\
    /* make the ray a unit vector */\
    float normFactor = 1.0f / sqrt((rx[vol_id] * rx[vol_id]) + (ry[vol_id] * ry[vol_id]) + (rz[vol_id] * rz[vol_id]));\
    rx[vol_id] *= normFactor;\
    ry[vol_id] *= normFactor;\
    rz[vol_id] *= normFactor;\
    /*\
    float tmp = 0.0f;\
    tmp += (rx[vol_id] * gVoxelElementSizeX[vol_id])*(rx[vol_id] * gVoxelElementSizeX[vol_id]);\
    tmp += (ry[vol_id] * gVoxelElementSizeY[vol_id])*(ry[vol_id] * gVoxelElementSizeY[vol_id]);\
    tmp += (rz[vol_id] * gVoxelElementSizeZ[vol_id])*(rz[vol_id] * gVoxelElementSizeZ[vol_id]);\
    volume_normalization_factor[vol_id] = sqrtf(tmp);*/\
} while (0)

#if NUM_VOLUMES == 1
#define CALCULATE_RAYS do {\
    CALCULATE_RAY_FOR_VOL(0);\
} while (0)
#elif NUM_VOLUMES == 2
#define CALCULATE_RAYS do {\
    CALCULATE_RAY_FOR_VOL(0);\
    CALCULATE_RAY_FOR_VOL(1);\
} while (0)
#elif NUM_VOLUMES == 3
#define CALCULATE_RAYS do {\
    CALCULATE_RAY_FOR_VOL(0);\
    CALCULATE_RAY_FOR_VOL(1);\
    CALCULATE_RAY_FOR_VOL(2);\
} while (0)
#else
#define CALCULATE_RAYS do {\
    fprintf(stderr, "CALCULATE_RAYS not supported for NUM_VOLUMES outside [1, 3]");\
} while (0)
#endif

#define CALCULATE_ALPHAS_FOR_VOL(i) do{\
    minAlpha[i] = 0;\
    maxAlpha[i] = INFINITY;\
    do_trace[i] = 1;\
\
    if (0.0f != rx[i]) {\
        float reci = 1.0f / rx[i];\
        float alpha0 = (gVolumeEdgeMinPointX[i] - sx[i]) * reci;\
        float alpha1 = (gVolumeEdgeMaxPointX[i] - sx[i]) * reci;\
        minAlpha[i] = fmin(alpha0, alpha1);\
        maxAlpha[i] = fmax(alpha0, alpha1);\
    } else if (gVolumeEdgeMinPointX[i] > sx[i] || sx[i] > gVolumeEdgeMaxPointX[i]) {\
        do_trace[i] = 0;\
    }\
\
    if (do_trace[i] && (0.0f != ry[i])) {\
        float reci = 1.0f / ry[i];\
        float alpha0 = (gVolumeEdgeMinPointY[i] - sy[i]) * reci;\
        float alpha1 = (gVolumeEdgeMaxPointY[i] - sy[i]) * reci;\
        minAlpha[i] = fmax(minAlpha[i], fmin(alpha0, alpha1));\
        maxAlpha[i] = fmin(maxAlpha[i], fmax(alpha0, alpha1));\
    } else if (gVolumeEdgeMinPointY[i] > sy[i] || sy[i] > gVolumeEdgeMaxPointY[i]) {\
        do_trace[i] = 0;\
    }\
\
    if (do_trace[i] && (0.0f != rz[i]))  {\
        float reci = 1.0f / rz[i];\
        float alpha0 = (gVolumeEdgeMinPointZ[i] - sz[i]) * reci;\
        float alpha1 = (gVolumeEdgeMaxPointZ[i] - sz[i]) * reci;\
        minAlpha[i] = fmax(minAlpha[i], fmin(alpha0, alpha1));\
        maxAlpha[i] = fmin(maxAlpha[i], fmax(alpha0, alpha1));\
    } else if (gVolumeEdgeMinPointZ > sz || sz > gVolumeEdgeMaxPointZ) {\
        do_trace[i] = 0;\
    }\
    globalMinAlpha = fmin(minAlpha[i], globalMinAlpha);\
    globalMaxAlpha = fmax(maxAlpha[i], globalMaxAlpha);\
} while (0)

#if NUM_VOLUMES == 1
#define CALCULATE_ALPHAS do {\
    CALCULATE_ALPHAS_FOR_VOL(0);\
} while (0)
#elif NUM_VOLUMES == 2
#define CALCULATE_ALPHAS do {\
    CALCULATE_ALPHAS_FOR_VOL(0);\
    CALCULATE_ALPHAS_FOR_VOL(1);\
} while (0)
#elif NUM_VOLUMES == 3
#define CALCULATE_ALPHAS do {\
    CALCULATE_ALPHAS_FOR_VOL(0);\
    CALCULATE_ALPHAS_FOR_VOL(1);\
    CALCULATE_ALPHAS_FOR_VOL(2);\
} while (0)
#else
#define CALCULATE_ALPHAS do {\
    fprintf(stderr, "CALCULATE_ALPHAS not supported for NUM_VOLUMES outside [1, 3]");\
} while (0)
#endif

#define GET_PRIORITY_AT_ALPHA(curr_priority, n_vols_at_curr_priority, alpha) do {\
    curr_priority = NUM_VOLUMES;\
    n_vols_at_curr_priority = 0;\
    for (int i = 0; i < NUM_VOLUMES; i++) {\
        if (0 == do_trace[i]) { continue; }\
        if ((alpha < minAlpha[i]) || (alpha > maxAlpha[i])) { continue; }\
\
        if (priority[i] < curr_priority) {\
            curr_priority = priority[i];\
            n_vols_at_curr_priority = 1;\
        } else if (priority[i] == curr_priority) {\
            n_vols_at_curr_priority ++;\
        }\
    }\
} while (0)

extern "C" {
    __global__  void projectKernel(
        int out_width, // width of the output image
        int out_height, // height of the output image
        float step,
        int *priority, // volumes with smaller priority-ID have higher priority when determining which volume we are in
        float *gVolumeEdgeMinPointX, // one value for each of the NUM_VOLUMES volumes
        float *gVolumeEdgeMinPointY,
        float *gVolumeEdgeMinPointZ,
        float *gVolumeEdgeMaxPointX,
        float *gVolumeEdgeMaxPointY,
        float *gVolumeEdgeMaxPointZ,
        float *gVoxelElementSizeX, // one value for each of the NUM_VOLUMES volumes
        float *gVoxelElementSizeY,
        float *gVoxelElementSizeZ,
        float *sx, // x-coordinate of source point for rays in world-space
        float *sy, // one value for each of the NUM_VOLUMES volumes
        float *sz,
        float *rt_kinv, // (NUM_VOLUMES, 3, 3) array giving the image-to-world-ray transform for each volume
        int n_bins, // the number of spectral bins
        float *energies, // 1-D array -- size is the n_bins. Units: [keV]
        float *pdf, // 1-D array -- probability density function over the energies
        float *absorb_coef_table, // flat [n_bins x NUM_MATERIALS] table that represents
                        // the precomputed get_absorption_coef values.
                        // index into the table as: table[bin * NUM_MATERIALS + mat]
        float *intensity, // flat array, with shape (out_height, out_width).
        float *photon_prob, // flat array, with shape (out_height, out_width).
        int offsetW,
        int offsetH)
    {
        // The output image has the following coordinate system, with cell-centered sampling.
        // y is along the fast axis (columns), x along the slow (rows).
        // Each point has NUM_MATERIALS elements at it.
        // 
        //      x -->
        //    y *---------------------------*
        //    | |                           |
        //    V |                           |
        //      |        output image       |
        //      |                           |
        //      |                           |
        //      *---------------------------*
        // 
        //
        int udx = threadIdx.x + (blockIdx.x + offsetW) * blockDim.x; // index into output image width
        int vdx = threadIdx.y + (blockIdx.y + offsetH) * blockDim.y; // index into output image height

        // if the current point is outside the output image, no computation needed
        if (udx >= out_width || vdx >= out_height)
            return;

        if ((0 == udx) && (0 == vdx)) {
            for (int i = 0; i < NUM_VOLUMES; i++) {
                printf("priority #%d: %d\n", i, priority[i]);
            }
        }

        // cell-centered sampling point corresponding to pixel index, in index-space.
        float u = (float) udx + 0.5;
        float v = (float) vdx + 0.5;

        // Vector in voxel-space along ray from source-point to pixel at [u,v] on the detector plane.
        float rx[NUM_VOLUMES];
        float ry[NUM_VOLUMES];
        float rz[NUM_VOLUMES];
        float volume_normalization_factor[NUM_VOLUMES];
        CALCULATE_RAYS;

        for (int i = 0; i < NUM_VOLUMES; i++) {
            float tmp = 0.0f;
            tmp += (rx[i] * gVoxelElementSizeX[i])*(rx[i] * gVoxelElementSizeX[i]);
            tmp += (ry[i] * gVoxelElementSizeY[i])*(ry[i] * gVoxelElementSizeY[i]);
            tmp += (rz[i] * gVoxelElementSizeZ[i])*(rz[i] * gVoxelElementSizeZ[i]);
            volume_normalization_factor[i] = sqrtf(tmp);
            if ((0==udx) && (0==vdx)) printf("volume_normalization_factor[%d]: %f\n", i, volume_normalization_factor[i]);
        }

        // calculate projections
        // Part 1: compute alpha value at entry and exit point of the volume on either side of the ray.
        // minAlpha: the distance from source point to volume entry point of the ray.
        // maxAlpha: the distance from source point to volume exit point of the ray.
        float minAlpha[NUM_VOLUMES];
        float maxAlpha[NUM_VOLUMES];
        int do_trace[NUM_VOLUMES]; // for each volume, whether or not to perform the ray-tracing
        float globalMinAlpha = INFINITY; // the smallest of all the minAlpha's
        float globalMaxAlpha = 0.0f; // the largest of all the maxAlpha's
        CALCULATE_ALPHAS;

        if ((600 == udx) && (400 == vdx)) {
            for (int i = 0; i < NUM_VOLUMES; i++) {
                printf("minAlpha[%d]=%f, maxAlpha[%d]=%f\n", i, minAlpha[i], i, maxAlpha[i]);
            }
            printf("globalMinAlpha=%f, globalMaxAlpha=%f\n", globalMinAlpha, globalMaxAlpha);
        }

        // we start not at the exact entry point 
        // => we can be sure to be inside the volume
        // (this is commented out intentionally, seemingly)
        //for (int i = 0; i < NUM_VOLUMES; i++) {
        //    minAlpha[i] += step * 0.5f;
        //}

        // Determine whether to do any ray-tracing at all. 
        // BLAH Use [out_width] as variable because it is no longer needed
        for (int i = 0; i < NUM_VOLUMES; i++) {
            if (do_trace[i]) { break; }
            else if ((NUM_VOLUMES - 1) == i) { return; }
        }
        
        // Part 2: Cast ray if it intersects the volume

        // material projection-output channels
        float area_density[NUM_MATERIALS]; 

        // initialize the projection-output to 0.
        for (int m = 0; m < NUM_MATERIALS; m++) {
            area_density[m] = 0;
        }

        float px, py, pz; // voxel-space point -- temporary storage
        float alpha; // distance along ray (alpha = globalMinAlpha + step * t)
        float boundary_factor; // factor to multiply at boundary
        int curr_priority; // the priority at the location
        int n_vols_at_curr_priority; // how many volumes to consider at the location
        float adiatl[NUM_MATERIALS]; // area_density increment at this location

        for (alpha = globalMinAlpha; alpha < globalMaxAlpha; alpha += step) {
            // Determine priority at the location -- TODO: macro-ify
            curr_priority = NUM_VOLUMES;
            n_vols_at_curr_priority = 0;
            for (int i = 0; i < NUM_VOLUMES; i++) {
                if (0 == do_trace[i]) { continue; }
                if ((alpha < minAlpha[i]) || (alpha > maxAlpha[i])) { continue; }
        
                if (priority[i] < curr_priority) {
                    curr_priority = priority[i];
                    n_vols_at_curr_priority = 1;
                } else if (priority[i] == curr_priority) {
                    n_vols_at_curr_priority ++;
                }
            }
            if ((n_vols_at_curr_priority <= 0)) { 
                printf("ERROR at alpha=%f. No volumes at current priority (%d) detected\n", alpha, curr_priority);
                for (int i = 0; i < NUM_VOLUMES; i++) {
                    if (0 == i) GET_POSITION_FOR_VOL(0);
                    else if (1 == i) GET_POSITION_FOR_VOL(1);
                    else if (2 == i) GET_POSITION_FOR_VOL(2);
                    else { printf("invalid volume ID\n"); break; }
                    printf("\tvolume#%d (priority=%d) position: %f, %f, %f\n", i, priority[i], px, py, pz);
                }
            }
            float weight = 1.0f / ((float) n_vols_at_curr_priority); // each volume contributes WEIGHT to the area_density
            if (0 == n_vols_at_curr_priority) printf("WARNING WARNING WARNING: dividing by (n_vols_at_curr_priority == 0)\n");
            if ((1 == NUM_VOLUMES) && (1.0f != weight)) printf("WARNING WARNING WARNING: improper 'weight' value (%f)\n", weight);

            // For the entry boundary, multiply by 0.5. That is, for the initial interpolated value,
            // only a half step-size is considered in the computation. For the second-to-last interpolation
            // point, also multiply by 0.5, since there will be a final step at the globalMaxAlpha boundary.
            boundary_factor = ((alpha <= globalMinAlpha) || (alpha + step >= globalMaxAlpha)) ? 0.5f : 1.0f;
            
            INTERPOLATE(boundary_factor);
            for (int m = 0; m < NUM_MATERIALS; m++) {
                area_density[m] += adiatl[m] * weight;
                if (adiatl[m] != adiatl[m]) {
                    //printf("adiatl[%d] is NaN\n", m);
                    //for (int i = 0; i < 12345678; i++) ;
                }
            }
        }

        // Scaling by step
        for (int m = 0; m < NUM_MATERIALS; m++) {
            area_density[m] *= step;

            if (area_density[m] != area_density[m]) {
                //printf("mat: %d, NaN\n", m);
            }
        }

        // Last segment of the line
        /*Bif (area_density[0] > 0.0f) {
            // ERROR IN HERE
            float lastStepsize = globalMaxAlpha - alpha;

            // Determine priority at the location -- TODO: macro-ify
            curr_priority = NUM_VOLUMES;
            n_vols_at_curr_priority = 0;
            for (int i = 0; i < NUM_VOLUMES; i++) {
                if ((alpha < minAlpha[i]) || (alpha > maxAlpha[i])) { continue; }
                
                if (priority[i] < curr_priority) {
                    curr_priority = priority[i];
                    n_vols_at_curr_priority = 1;
                } else if (priority[i] == curr_priority) {
                    n_vols_at_curr_priority ++;
                }
            }
            //Bif (n_vols_at_curr_priority <= 0) { printf("ERROR at alpha=%f. No volumes at current priority (%d) detected\n", alpha, curr_priority); }
            float weight = 1.0f / ((float) n_vols_at_curr_priority); // each volume contributes WEIGHT to the area_density

            // Scaled last step interpolation (something weird?)
            INTERPOLATE(0.5f * lastStepsize);
            if (0 > n_vols_at_curr_priority) {
                printf("ERROR: n_vols_at_curr_priority < 0: is %d\n", n_vols_at_curr_priority);
            } else if (0 == n_vols_at_curr_priority) {
                /*for (int m = 0; m < NUM_MATERIALS; m++) {
                    if (0 != adiatl[m]){
                        int bytes;
                        memcpy(&bytes, &adiatl[m], sizeof(int));
                        char bits[33];
                        for (int i = 0; i < 32; i++) {
                            bits[32 - i - 1] = '0' + (bytes & 0x00000001);
                            bytes >>= 1;
                        }
                        bits[33] = '\0';
                        printf(
                            "n_vols_at_curr_priority == 0. mat=%d, adiatl[mat]=%1.10e != 0. <- bad\n"
                            "\tadiatl[mat] = 0b%s\n",
                            m, adiatl[m], bits
                        );
                    }
                }*/
                /*Bweight = 1.0f;
            }
            for (int m = 0; m < NUM_MATERIALS; m++) {
                area_density[m] += adiatl[m] * weight;
            }

            alpha -= step;

            // Determine priority at the location -- TODO: macro-fy
            curr_priority = NUM_VOLUMES;
            n_vols_at_curr_priority = 0;
            for (int i = 0; i < NUM_VOLUMES; i++) {
                if ((alpha < minAlpha[i]) || (alpha > maxAlpha[i])) { continue; }

                if (priority[i] < curr_priority) {
                    curr_priority = priority[i];
                    n_vols_at_curr_priority = 1;
                } else if (priority[i] == curr_priority) {
                    n_vols_at_curr_priority ++;
                }
            }
            //Bif (n_vols_at_curr_priority <= 0) { printf("ERROR at alpha=%f. No volumes at current priority (%d) detected\n", alpha, curr_priority); }
            weight = 1.0f / ((float) n_vols_at_curr_priority); // each volume contributes WEIGHT to the area_density

            // The last segment of the line integral takes care of the varying length.
            INTERPOLATE(0.5f * lastStepsize);
            for (int m = 0; m < NUM_MATERIALS; m++) {
                area_density[m] += adiatl[m] * weight;
            }
        }*/

        // Convert to centimeters
        for (int m = 0; m < NUM_MATERIALS; m++) {
            area_density[m] /= 10.0f;
        }

        /*if (area_density[1] == 0.0f) {
            printf("pixel [%d, %d]. Channel[0]: %f, Channel[2]: %f\n", udx, vdx, area_density[0], area_density[2]);
        }*/
        if ((area_density[0] != 0.0f) || (area_density[1] != 0.0f) || (area_density[2] != 0.0f)) {
            //printf("pixel [%d, %d]. Channel[0]: %1.16e, Channel[1]: %1.16e, Channel[2]: %1.16e\n", udx, vdx, area_density[0], area_density[1], area_density[1]);
        }

        /*
        for (int i = 0; i < NUM_VOLUMES; i++) {
            // Trapezoidal rule (interpolating function = piecewise linear func)
            float px, py, pz; // voxel-space point
            int t; // number of steps along ray //
            float alpha; // distance along ray (alpha = minAlpha + step * t)
            float boundary_factor; // factor to multiply at the boundary.
        
            // Sample the points along the ray at the entrance boundary of the volume and the mid segments.
            for (t = 0, alpha = minAlpha[i]; alpha < maxAlpha[i]; t++, alpha += step)
            {
                // Get the current sample point in the volume voxel-space.
                // In CUDA, voxel centeras are located at (xx.5, xx.5, xx.5), whereas SwVolume has voxel centers at integers.
                px = sx[i] + alpha * rx[i] - gVolumeEdgeMinPointX[i];
                py = sy[i] + alpha * ry[i] - gVolumeEdgeMinPointY[i];
                pz = sz[i] + alpha * rz[i] - gVolumeEdgeMinPointZ[i];
        
                // For the entry boundary, multiply by 0.5 (this is the t == 0 check). That is, for the initial interpolated value,
                // only a half step-size is considered in the computation.
                // For the second-to-last interpolation point, also multiply by 0.5, since there will be a final step at the maxAlpha boundary.
                boundary_factor = (t == 0 || alpha + step >= maxAlpha[i]) ? 0.5 : 1.0;
        
                // Perform the interpolation. This involves the variables: area_density, idx, px, py, pz, and volume. 
                // It is done for each segmentation. 
                INTERPOLATE_FOR_VOL(boundary_factor, i);
            }
        
            // Scaling by step; 
            for (int m = 0; m < NUM_MATERIALS; m++) {
                area_density[m] *= step;
            }
        
            // Last segment of the line 
            if (area_density[0] > 0.0f) {
                alpha -= step;
                float lastStepsize = maxAlpha[i] - alpha;
                // scaled last step interpolation (something weird?) 
                INTERPOLATE_FOR_VOL(0.5 * lastStepsize, i);
                // The last segment of the line integral takes care of the varying length.
                px = sx[i] + alpha * rx[i] - gVolumeEdgeMinPointX[i];
                py = sy[i] + alpha * ry[i] - gVolumeEdgeMinPointY[i];
                pz = sz[i] + alpha * rz[i] - gVolumeEdgeMinPointZ[i];
                // interpolation 
                INTERPOLATE_FOR_VOL(0.5 * lastStepsize, i);
            }
        
            // normalize output value to world coordinate system units 
            for (int m = 0; m < NUM_MATERIALS; m++) {
                area_density[m] *= sqrt((rx[i] * gVoxelElementSizeX[i])*(rx[i] * gVoxelElementSizeX[i]) + (ry[i] * gVoxelElementSizeY[i])*(ry[i] * gVoxelElementSizeY[i]) + (rz[i] * gVoxelElementSizeZ[i])*(rz[i] * gVoxelElementSizeZ[i]));
                
                total_area_density[m] += area_density[m];
            }
        }
        */

        /* Up to this point, we have accomplished the original projectKernel functionality.
         * The next steps to do are combining the forward_projections dictionary-ization and 
         * the mass_attenuation computation
         */

        // forward_projections dictionary-ization is implicit.

        // flat index to pixel in *intensity and *photon_prob
        int img_dx = (udx * out_height) + vdx;

        // zero-out intensity and photon_prob
        intensity[img_dx] = 0;
        photon_prob[img_dx] = 0;

        // MASS ATTENUATION COMPUTATION

        /**
         * EXPLANATION OF THE PHYSICS/MATHEMATICS
         * 
         *      The mass attenuation coefficient (found in absorb_coef_table) is: \mu / \rho, where
         * \mu is the linear attenuation coefficient, and \rho is the mass density.  \mu has units of
         * inverse length, and \rho has units of mass/volume, so the mass attenuation coefficient has
         * units of [cm^2 / g]
         *      area_density[m] is the product of [linear distance of the ray through material 'm'] and 
         * [density of the material].  Accordingly, area_density[m] has units of [g / cm^2].
         *
         * The mass attenuation code uses the Beer-Lambert law:
         *
         *      I = I_{0} exp[-(\mu / \rho) * \rho * d]
         *
         * where I_{0} is the initial intensity, (\mu / \rho) is the mass attenuation coefficient, 
         * \rho is the density, and d is the length of the ray passing through the material.  Note 
         * that the product (\rho * d), also known as the 'area density' is the quantity area_density[m].
         *      Because we are attenuating multiple materials, the exponent that we use for the 
         * Beer-Lambert law is the sum of the (\mu_{mat} / \rho_{mat}) * (\rho_{mat} * d_{mat}) for
         * each material 'mat'.
         *
         *      The above explains the calculation up to and including 
         *              '____ = expf(-1 * beer_lambert_exp)',
         * but does not yet explain the remaining calculation.  The remaining calculation serves to 
         * approximate the workings of a pixel in the dectector:
         *      
         *      pixelReading = \sum_{E} attenuatedBeamStrength[E] * E * p(E)
         *
         * where attenuatedBeamStrength follows the Beer-Lambert law as above, E is the energies of
         * the spectrum, and p(E) is the PDF of the spectrum.
         *      Note also that the Beer-Lambert law deals with the quantity 'intensity', which is 
         * related to the power transmitted through [unit area perpendicular to the direction of travel].
         * Since the intensities mentioned in the Beer-Lambert law are proportional to 1/[unit area], we
         * can replace the "intensity" calcuation with simply the energies involved.  Later conversion to 
         * other physical quanities can be done outside of the kernel.
         */
        for (int bin = 0; bin < n_bins; bin++) {
            float beer_lambert_exp = 0.0f;
            for (int m = 0; m < NUM_MATERIALS; m++) {
                beer_lambert_exp += area_density[m] * absorb_coef_table[bin * NUM_MATERIALS + m];
            }
            float photon_prob_tmp = expf(-1.f * beer_lambert_exp) * pdf[bin]; // dimensionless value

            photon_prob[img_dx] += photon_prob_tmp;
            intensity[img_dx] += energies[bin] * photon_prob_tmp; // units: [keV] per unit photon to hit the pixel
        }

        return;
    }
}
    
