#include <stdio.h>
#include <cubicTex3D.cu>

#include "project_kernel_multi_data.cu"

extern "C" {
    // TODO: refactor to take advantage of parallelization
    __device__ void resample_megavolume(
        int inp_priority[NUM_VOLUMES],
        int inp_voxelBoundX[NUM_VOLUMES], // number of voxels in x direction for each volume
        int inp_voxelBoundY[NUM_VOLUMES],
        int inp_voxelBoundZ[NUM_VOLUMES],
        //float inp_gVoxelElementSizeX[NUM_VOLUMES], // voxel size for input volumes, in world coordinates
        //float inp_gVoxelElementSizeY[NUM_VOLUMES],
        //float inp_gVoxelElementSizeZ[NUM_VOLUMES],
        float inp_ijk_from_world[9 * NUM_VOLUMES], // ijk_from_world transforms for input volumes TODO: is each transform 3x3?
        float megaMinX, // bounding box for output megavolume, in world coordinates
        float megaMinY,
        float megaMinZ,
        float megaMaxX,
        float megaMaxY,
        float megaMaxZ,
        float megaVoxelSizeX, // voxel size for output megavolume, in world coordinates
        float megaVoxelSizeY,
        float megaVoxelSizeZ,
        //float mega_ijk_from_world[9], // TODO: is each transform 3x3? --- this never actually gets used
        int mega_x_len, // the (exclusive, upper) array index bound of the megavolume
        int mega_y_len,
        int mega_z_len,
        float *output_density, // volume-sized array
        char *output_mat_id // volume-sized array to hold the material IDs of the voxels
    ) {
        /*
         * Sample in voxel centers.
         * 
         * Loop keeps track of {x,y,z} position in world coord.s as well as IJK indices for megavolume voxels.
         * The first voxel has IJK indices (0,0,0) and is centered at (minX + 0.5 * voxX, minY + 0.5 * voxY, minZ + 0.5 * voxZ)
         *
         * The upper bound of the loop checking for:
         *       {x,y,z} <= megaMax{X,Y,Z}
         * is sufficient because the preprocessing of the boudning box ensured that the voxels fit neatly into the bounding box
         */

        // local storage to store the results of the tex3D calls.
        // As a switch, we rely on the fact that the results of the tex3D calls should never be negative
        float density_sample[NUM_VOLUMES];
        // local storage to store the results of the cubicTex3D calls
        float mat_sample[NUM_VOLUMES][NUM_MATERIALS];
        
        for (float x = megaMinX + (megaVoxelSizeX * 0.5), int x_ind = 0; x <= megaMaxX; x += megaVoxelSizeX, x_ind++) {
            for (float y = megaMinY + (megaVoxelSizeY * 0.5), int y_ind = 0; y <= megaMaxY; y += megaVoxelSizeY, y_ind++) {
                for (float z = megaMinZ + (megaVoxelSizeZ * 0.5), int z_ind = 0; z <= megaMaxZ; z += megaVoxelSizeZ, z_ind++) {
                    // for each volume, check whether we are inside its bounds
                    int curr_priority = NUM_VOLUMES;

                    for (int i = 0; i < NUM_VOLUMES; i++) {
                        density_sample[i] = -1.0f; // "reset" this volume's sample

                        int offset = 9 * i;
                        float inp_x = (inp_ijk_from_world[offset + 0] * x) + (inp_ijk_from_world[offset + 1] * y) + (inp_ijk_from_world[offset + 2] * z);
                        if ((inp_x < 0.0) || (inp_x >= inp_voxelBoundX[i])) continue; // TODO: make sure this behavior agrees with the behavior of ijk_from_world transforms

                        float inp_y = (inp_ijk_from_world[offset + 3] * x) + (inp_ijk_from_world[offset + 4] * y) + (inp_ijk_from_world[offset + 5] * z);
                        if ((inp_y < 0.0) || (inp_y >= inp_voxelBoundY[i])) continue;

                        float inp_z = (inp_ijk_from_world[offset + 6] * x) + (inp_ijk_from_world[offset + 7] * y) + (inp_ijk_from_world[offset + 8] * z);
                        if ((inp_z < 0.0) || (inp_z >= inp_voxelBoundZ[i])) continue;

                        if (inp_priority[i] < curr_priority) curr_priority = inp_priority[i];
                        else if (inp_priority[i] > curr_priority) continue;

                        // TODO: macro-ify these calls to texture interpolation
                        density_sample[i] = tex3D(VOLUME(i), inp_x, inp_y, inp_z);
                        for (int m = 0; m < NUM_MATERIALS; m++) {
                            mat_sample[i][m] = cubicTex3D(SEG(i, m), inp_x, inp_y, inp_z);
                        }
                    }

                    int output_idx = x_ind + (y_ind * mega_x_len) + (z_ind * mega_x_len * mega_y_len);
                    if (NUM_VOLUMES == curr_priority) {
                        // no input volumes at the current point
                        output_density[output_idx] = 0.0f;
                        output_mat_id[output_idx] = NUM_MATERIALS; // out of range for mat id, so indicates no material
                    } else {
                        // for averaging the densities of the volumes to "mix"
                        int n_vols_at_curr_priority = 0;
                        float total_density = 0.0f;

                        // for determining the material most 
                        float total_mat_seg[NUM_MATERIALS];
                        for (int m = 0; m < NUM_MATERIALS; m++) {
                            total_mat_seg[m] = 0.0f;
                        }

                        for (int i = 0; i < NUM_VOLUMES; i++) {
                            if (curr_priority == inp_priority[i]) {
                                n_vols_at_curr_priority++;
                                total_density += density_sample[i];

                                for (int m = 0; m < NUM_MATERIALS; m++) {
                                    total_mat_seg[m] = mat_sample[i][m];
                                }
                            }
                        }

                        int mat_id = NUM_MATERIALS;
                        float highest_mat_seg = 0.0f;
                        for (int m = 0; m < NUM_MATERIALS; m++) {
                            if (total_mat_seg[m] > highest_mat_seg) {
                                mat_id = m;
                                highest_mat_seg = total_mat_seg[m];
                            }
                        }

                        output_density[output_idx] = total_density / ((float) n_vols_at_curr_priority);
                        output_mat_id[output_idx] = mat_id;
                    }
                }
            }
        }

        return;
    }
}