#include <stdio.h>
#include <cubicTex3D.cu>

#include "project_kernel_data.cu"

#define UPDATE(multiplier, vol_id, mat_id) do {\
    /* param 'weight' is the 1.0f/(number of volumes at curr_priority) */\
    /*adiatl[(mat_id)] = (multiplier) * tex3D(VOLUME(vol_id), px[vol_id], py[vol_id], pz[vol_id]) * seg_at_alpha[vol_id][mat_id] * volume_normalization_factor[vol_id];*/\
    area_density[(mat_id)] += (multiplier) * tex3D(VOLUME(vol_id), px[vol_id], py[vol_id], pz[vol_id]) * seg_at_alpha[vol_id][mat_id] * volume_normalization_factor[vol_id] * weight;\
    output_for_vol[(vol_id)][(mat_id)] += (multiplier) * tex3D(VOLUME(vol_id), px[vol_id], py[vol_id], pz[vol_id]) * seg_at_alpha[vol_id][mat_id] * volume_normalization_factor[vol_id];\
} while (0)

#define GET_POSITION_FOR_VOL(vol_id) do {\
    /* Get the current sample point in the volume voxel-space. */\
    /* In CUDA, voxel centers are located at (xx.5, xx.5, xx.5), whereas SwVolume has voxel centers at integers. */\
    px[vol_id] = sx[vol_id] + alpha * rx[vol_id] - gVolumeEdgeMinPointX[vol_id];\
    py[vol_id] = sy[vol_id] + alpha * ry[vol_id] - gVolumeEdgeMinPointY[vol_id];\
    pz[vol_id] = sz[vol_id] + alpha * rz[vol_id] - gVolumeEdgeMinPointZ[vol_id];\
} while (0)

#define LOAG_SEGS_FOR_VOL_MAT(vol_id, mat_id) do {\
    seg_at_alpha[vol_id][mat_id] = round(cubicTex3D(SEG(vol_id, mat_id), px[vol_id], py[vol_id], pz[vol_id]));\
    /*if (seg_at_alpha[vol_id][mat_id] > 0.0f) {\
        printf("at position {%f, %f, %f}, seg_at_alpha[%d][%d]=%f > 0.0f\n", px[vol_id], py[vol_id], pz[vol_id], vol_id, mat_id, seg_at_alpha[vol_id][mat_id]);\
    }*/\
} while (0)

#if NUM_MATERIALS == 1
#define LOAD_SEGS_FOR_VOL(vol_id) do {\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 0);\
} while (0)
#elif NUM_MATERIALS == 2
#define LOAD_SEGS_FOR_VOL(vol_id) do {\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 0);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 1);\
} while (0)
#elif NUM_MATERIALS == 3
#define LOAD_SEGS_FOR_VOL(vol_id) do {\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 0);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 1);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 2);\
} while (0)
#elif NUM_MATERIALS == 4
#define LOAD_SEGS_FOR_VOL(vol_id) do {\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 0);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 1);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 2);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 3);\
} while (0)
#elif NUM_MATERIALS == 5
#define LOAD_SEGS_FOR_VOL(vol_id) do {\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 0);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 1);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 2);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 3);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 4);\
} while (0)
#elif NUM_MATERIALS == 6
#define LOAD_SEGS_FOR_VOL(vol_id) do {\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 0);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 1);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 2);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 3);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 4);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 5);\
} while (0)
#elif NUM_MATERIALS == 7
#define LOAD_SEGS_FOR_VOL(vol_id) do {\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 0);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 1);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 2);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 3);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 4);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 5);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 6);\
} while (0)
#elif NUM_MATERIALS == 8
#define LOAD_SEGS_FOR_VOL(vol_id) do {\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 0);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 1);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 2);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 3);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 4);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 5);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 6);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 7);\
} while (0)
#elif NUM_MATERIALS == 9
#define LOAD_SEGS_FOR_VOL(vol_id) do {\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 0);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 1);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 2);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 3);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 4);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 5);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 6);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 7);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 8);\
} while (0)
#elif NUM_MATERIALS == 10
#define LOAD_SEGS_FOR_VOL(vol_id) do {\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 0);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 1);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 2);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 3);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 4);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 5);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 6);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 7);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 8);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 9);\
} while (0)
#elif NUM_MATERIALS == 11
#define LOAD_SEGS_FOR_VOL(vol_id) do {\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 0);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 1);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 2);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 3);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 4);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 5);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 6);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 7);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 8);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 9);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 10);\
} while (0)
#elif NUM_MATERIALS == 12
#define LOAD_SEGS_FOR_VOL(vol_id) do {\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 0);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 1);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 2);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 3);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 4);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 5);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 6);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 7);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 8);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 9);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 10);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 11);\
} while (0)
#elif NUM_MATERIALS == 13
#define LOAD_SEGS_FOR_VOL(vol_id) do {\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 0);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 1);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 2);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 3);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 4);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 5);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 6);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 7);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 8);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 9);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 10);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 11);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 12);\
} while (0)
#elif NUM_MATERIALS == 14
#define LOAD_SEGS_FOR_VOL(vol_id) do {\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 0);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 1);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 2);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 3);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 4);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 5);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 6);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 7);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 8);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 9);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 10);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 11);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 12);\
    LOAG_SEGS_FOR_VOL_MAT(vol_id, 13);\
} while (0)
#else
#define LOAD_SEGS_FOR_VOL(vol_id) do {\
    fprintf(stderr, "NUM_MATERIALS not in [1, 14]");\
} while (0)
#endif

#if NUM_VOLUMES == 1
#define LOAD_SEGS_AT_ALPHA do {\
    if (do_trace[0]) { GET_POSITION_FOR_VOL(0); LOAD_SEGS_FOR_VOL(0); }\
} while (0)
#elif NUM_VOLUMES == 2
#define LOAD_SEGS_AT_ALPHA do {\
    if (do_trace[0]) {\
        GET_POSITION_FOR_VOL(0);\
        LOAD_SEGS_FOR_VOL(0);\
        /*int has_nonzero_seg = 0;\
        for (int __m = 0; __m < NUM_MATERIALS; __m++) {\
            if (seg_at_alpha[0][__m] > 0.0f) {\
                has_nonzero_seg = 1;\
                printf("at position {%f, %f, %f}, seg_at_alpha[%d][%d]=%f > 0.0f\n", px[0], py[0], pz[0], 0, __m, seg_at_alpha[0][__m]);\
                break;\
            }\
        }\
        if (!has_nonzero_seg) {\
           THIS NEVER TRIGGERED\
            printf("at position {%f, %f, %f}, no non-zero seg for volume0\n", px[0], py[0], pz[0]);\
        }*/\
    }\
    if (do_trace[1]) { GET_POSITION_FOR_VOL(1); LOAD_SEGS_FOR_VOL(1); }\
} while (0)
#elif NUM_VOLUMES == 3
#define LOAD_SEGS_AT_ALPHA do {\
    if (do_trace[0]) { GET_POSITION_FOR_VOL(0); LOAD_SEGS_FOR_VOL(0); }\
    if (do_trace[1]) { GET_POSITION_FOR_VOL(1); LOAD_SEGS_FOR_VOL(1); }\
    if (do_trace[2]) { GET_POSITION_FOR_VOL(2); LOAD_SEGS_FOR_VOL(2); }\
} while (0)
#else
#define LOAD_SEGS_AT_ALPHA do {\
    fprintf(stderr, "CALCULATE_RAYS not supported for NUM_VOLUMES outside [1, 3]");\
} while (0)
#endif

#if NUM_MATERIALS == 1
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    UPDATE(multiplier, vol_id, 0);\
} while (0)
#elif NUM_MATERIALS == 2
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    UPDATE(multiplier, vol_id, 0);\
    UPDATE(multiplier, vol_id, 1);\
} while (0)
#elif NUM_MATERIALS == 3
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    UPDATE(multiplier, vol_id, 0);\
    UPDATE(multiplier, vol_id, 1);\
    UPDATE(multiplier, vol_id, 2);\
} while (0)
#elif NUM_MATERIALS == 4
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    UPDATE(multiplier, vol_id, 0);\
    UPDATE(multiplier, vol_id, 1);\
    UPDATE(multiplier, vol_id, 2);\
    UPDATE(multiplier, vol_id, 3);\
} while (0)
#elif NUM_MATERIALS == 5
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    UPDATE(multiplier, vol_id, 0);\
    UPDATE(multiplier, vol_id, 1);\
    UPDATE(multiplier, vol_id, 2);\
    UPDATE(multiplier, vol_id, 3);\
    UPDATE(multiplier, vol_id, 4);\
} while (0)
#elif NUM_MATERIALS == 6
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
    UPDATE(multiplier, vol_id, 0);\
    UPDATE(multiplier, vol_id, 1);\
    UPDATE(multiplier, vol_id, 2);\
    UPDATE(multiplier, vol_id, 3);\
    UPDATE(multiplier, vol_id, 4);\
    UPDATE(multiplier, vol_id, 5);\
} while (0)
#elif NUM_MATERIALS == 7
#define INTERPOLATE_FOR_VOL(multiplier, vol_id) do {\
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
    if (do_trace[0] && (priority[0] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 0); }\
} while (0)
#elif NUM_VOLUMES == 2
#define INTERPOLATE(multiplier) do {\
    if (do_trace[0] && (priority[0] == curr_priority)) { /*printf("interp0: alpha=%f, pixel=[%d,%d]\n", alpha, udx, vdx);*/ INTERPOLATE_FOR_VOL(multiplier, 0); }\
    if (do_trace[1] && (priority[1] == curr_priority)) { /*printf("interp1: alpha=%f, pixel=[%d,%d]\n", alpha, udx, vdx);*/ INTERPOLATE_FOR_VOL(multiplier, 1); }\
} while (0)
#elif NUM_VOLUMES == 3
#define INTERPOLATE(multiplier) do {\
    if (do_trace[0] && (priority[0] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 0); }\
    if (do_trace[1] && (priority[1] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 1); }\
    if (do_trace[2] && (priority[2] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 2); }\
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
    \
    float tmp = 0.0f;\
    tmp += (rx[vol_id] * gVoxelElementSizeX[vol_id])*(rx[vol_id] * gVoxelElementSizeX[vol_id]);\
    tmp += (ry[vol_id] * gVoxelElementSizeY[vol_id])*(ry[vol_id] * gVoxelElementSizeY[vol_id]);\
    tmp += (rz[vol_id] * gVoxelElementSizeZ[vol_id])*(rz[vol_id] * gVoxelElementSizeZ[vol_id]);\
    volume_normalization_factor[vol_id] = sqrtf(tmp);\
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

#define GET_PRIORITY_AT_ALPHA do {\
    curr_priority = NUM_VOLUMES;\
    n_vols_at_curr_priority = 0;\
    for (int i = 0; i < NUM_VOLUMES; i++) {\
        if (0 == do_trace[i]) { continue; }\
        if ((alpha < minAlpha[i]) || (alpha > maxAlpha[i])) { continue; }\
        float total_seg = 0.0f;\
        for (int m = 0; m < NUM_MATERIALS; m++) {\
            total_seg += seg_at_alpha[i][m];\
            if (total_seg > 0.0f) { break; }\
        }\
        if (0.0f == total_seg) { continue; }\
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

        if (do_trace[1] && !do_trace[0]) {
            printf("Huh? tracing volume1 but not volume0. pixel: [%d,%d]\n", udx, vdx);
        }

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
        for (int i = 0; i < NUM_VOLUMES; i++) {
            if (do_trace[i]) { break; }
            else if ((NUM_VOLUMES - 1) == i) { return; }
        }
        
        // Part 2: Cast ray if it intersects the volume

        // material projection-output channels
        float area_density[NUM_MATERIALS]; 

        // initialize the projection-output to 0.
        for (int m = 0; m < NUM_MATERIALS; m++) {
            area_density[m] = 0.0f;
        }

        float px[NUM_VOLUMES]; // voxel-space point
        float py[NUM_VOLUMES];
        float pz[NUM_VOLUMES];
        float alpha; // distance along ray (alpha = globalMinAlpha + step * t)
        float boundary_factor; // factor to multiply at boundary
        int curr_priority; // the priority at the location
        int n_vols_at_curr_priority; // how many volumes to consider at the location
        float adiatl[NUM_MATERIALS]; // area_density increment at this location
        float seg_at_alpha[NUM_VOLUMES][NUM_MATERIALS];

        float output_for_vol[NUM_VOLUMES][NUM_MATERIALS];
        for (int i = 0; i < NUM_VOLUMES; i++) for (int m = 0; m < NUM_MATERIALS; m++) output_for_vol[i][m] = 0.0f;

        for (alpha = globalMinAlpha; alpha < globalMaxAlpha; alpha += step) {
            LOAD_SEGS_AT_ALPHA; // initializes p{x,y,z}[...] and seg_at_alpha[...][...]
            if (do_trace[0]) {
                for (int mat = 0; mat < NUM_MATERIALS; mat++) {
                    if (0.5f > seg_at_alpha[0][mat]) {
                        //Bprintf("alpha=%f, p={%f, %f, %f}, round(cubicTex3D(seg_0_%d, ...))=%.10e\n", alpha, px[0], py[0], pz[0], mat, seg_at_alpha[0][mat]);
                    }
                }
            }
            GET_PRIORITY_AT_ALPHA;
            if (0 == n_vols_at_curr_priority) {
                // Outside the bounds of all volumes to trace. Assume nominal density of air is 0.0f.
                // Thus, we don't need to add to area_density
                ;
            } else {
                float weight = 1.0f / ((float) n_vols_at_curr_priority); // each volume contributes WEIGHT to the area_density
                
                // For the entry boundary, multiply by 0.5. That is, for the initial interpolated value,
                // only a half step-size is considered in the computation. For the second-to-last interpolation
                // point, also multiply by 0.5, since there will be a final step at the globalMaxAlpha boundary.
                boundary_factor = ((alpha <= globalMinAlpha) || (alpha + step >= globalMaxAlpha)) ? 0.5f : 1.0f;

                INTERPOLATE(boundary_factor);
                for (int m = 0; m < NUM_MATERIALS; m++) {
                    //BAarea_density[m] += adiatl[m] * weight;
                }
            }
        }

        if ((area_density[0] > 0.0f) || (area_density[1] > 0.0f)) {
            /*Bprintf(
                "after loop: a_d[0]=%.6e, a_d[1]=%.6e\n"
                "\toutput_for_vol[%d][%d]=%.6e\n"
                "\toutput_for_vol[%d][%d]=%.6e\n"
                "\toutput_for_vol[%d][%d]=%.6e\n"
                "\toutput_for_vol[%d][%d]=%.6e\n",
                area_density[0], area_density[1],
                0, 0, output_for_vol[0][0],
                0, 1, output_for_vol[0][1],
                1, 0, output_for_vol[1][0],
                1, 1, output_for_vol[1][1]
            );*/
        }

        // Scaling by step
        for (int m = 0; m < NUM_MATERIALS; m++) {
            area_density[m] *= step;
        }

        // Last segment of the line
        if (area_density[0] > 0.0f) {
            alpha -= step;
            float lastStepsize = globalMaxAlpha - alpha;

            GET_PRIORITY_AT_ALPHA;
            if (0 == n_vols_at_curr_priority) {
                // Outside the bounds of all volumes to trace. Assume nominal density of air is 0.0f.
                // Thus, we don't need to add to area_density
                ;
            } else {
                float weight = 1.0f / ((float) n_vols_at_curr_priority); // each volume contributes WEIGHT to the area_density
                // Scaled last step interpolation (something weird?)
                INTERPOLATE(lastStepsize);
                for (int m = 0; m < NUM_MATERIALS; m++) {
                    //BAarea_density[m] += adiatl[m] * weight;
                }
            }
        }

        // Convert to centimeters
        for (int m = 0; m < NUM_MATERIALS; m++) {
            area_density[m] /= 10.0f;
        }

        /*if (area_density[1] == 0.0f) {
            printf("pixel [%d, %d]. Channel[0]: %f, Channel[2]: %f\n", udx, vdx, area_density[0], area_density[2]);
        }*/
        /*if ((area_density[0] != 0.0f) || (area_density[1] != 0.0f) || (area_density[2] != 0.0f)) {
            printf("pixel [%d, %d]. Channel[0]: %1.16e, Channel[1]: %1.16e, Channel[2]: %1.16e\n", udx, vdx, area_density[0], area_density[1], area_density[1]);
        }*/

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
    
