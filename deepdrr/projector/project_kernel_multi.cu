#include <stdio.h>
#include <cubicTex3D.cu>

#include "project_kernel_multi_data.cu"

#define UPDATE(multiplier, vol_id, mat_id) do {\
    /* param. weight is set to 1.0f / (float)n_vols_at_curr_priority */\
    area_density[(mat_id)] += (multiplier) * tex3D(VOLUME(vol_id), px[vol_id], py[vol_id], pz[vol_id]) * seg_at_alpha[vol_id][mat_id] * volume_normalization_factor[vol_id] * weight;\
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
    if (do_trace[0]) { GET_POSITION_FOR_VOL(0); LOAD_SEGS_FOR_VOL(0); }\
    if (do_trace[1]) { GET_POSITION_FOR_VOL(1); LOAD_SEGS_FOR_VOL(1); }\
} while (0)
#elif NUM_VOLUMES == 3
#define LOAD_SEGS_AT_ALPHA do {\
    if (do_trace[0]) { GET_POSITION_FOR_VOL(0); LOAD_SEGS_FOR_VOL(0); }\
    if (do_trace[1]) { GET_POSITION_FOR_VOL(1); LOAD_SEGS_FOR_VOL(1); }\
    if (do_trace[2]) { GET_POSITION_FOR_VOL(2); LOAD_SEGS_FOR_VOL(2); }\
} while (0)
#elif NUM_VOLUMES == 4
#define LOAD_SEGS_AT_ALPHA do {\
    if (do_trace[0]) { GET_POSITION_FOR_VOL(0); LOAD_SEGS_FOR_VOL(0); }\
    if (do_trace[1]) { GET_POSITION_FOR_VOL(1); LOAD_SEGS_FOR_VOL(1); }\
    if (do_trace[2]) { GET_POSITION_FOR_VOL(2); LOAD_SEGS_FOR_VOL(2); }\
    if (do_trace[3]) { GET_POSITION_FOR_VOL(3); LOAD_SEGS_FOR_VOL(3); }\
} while (0)
#elif NUM_VOLUMES == 5
#define LOAD_SEGS_AT_ALPHA do {\
    if (do_trace[0]) { GET_POSITION_FOR_VOL(0); LOAD_SEGS_FOR_VOL(0); }\
    if (do_trace[1]) { GET_POSITION_FOR_VOL(1); LOAD_SEGS_FOR_VOL(1); }\
    if (do_trace[2]) { GET_POSITION_FOR_VOL(2); LOAD_SEGS_FOR_VOL(2); }\
    if (do_trace[3]) { GET_POSITION_FOR_VOL(3); LOAD_SEGS_FOR_VOL(3); }\
    if (do_trace[4]) { GET_POSITION_FOR_VOL(4); LOAD_SEGS_FOR_VOL(4); }\
} while (0)
#elif NUM_VOLUMES == 6
#define LOAD_SEGS_AT_ALPHA do {\
    if (do_trace[0]) { GET_POSITION_FOR_VOL(0); LOAD_SEGS_FOR_VOL(0); }\
    if (do_trace[1]) { GET_POSITION_FOR_VOL(1); LOAD_SEGS_FOR_VOL(1); }\
    if (do_trace[2]) { GET_POSITION_FOR_VOL(2); LOAD_SEGS_FOR_VOL(2); }\
    if (do_trace[3]) { GET_POSITION_FOR_VOL(3); LOAD_SEGS_FOR_VOL(3); }\
    if (do_trace[4]) { GET_POSITION_FOR_VOL(4); LOAD_SEGS_FOR_VOL(4); }\
    if (do_trace[5]) { GET_POSITION_FOR_VOL(5); LOAD_SEGS_FOR_VOL(5); }\
} while (0)
#elif NUM_VOLUMES == 7
#define LOAD_SEGS_AT_ALPHA do {\
    if (do_trace[0]) { GET_POSITION_FOR_VOL(0); LOAD_SEGS_FOR_VOL(0); }\
    if (do_trace[1]) { GET_POSITION_FOR_VOL(1); LOAD_SEGS_FOR_VOL(1); }\
    if (do_trace[2]) { GET_POSITION_FOR_VOL(2); LOAD_SEGS_FOR_VOL(2); }\
    if (do_trace[3]) { GET_POSITION_FOR_VOL(3); LOAD_SEGS_FOR_VOL(3); }\
    if (do_trace[4]) { GET_POSITION_FOR_VOL(4); LOAD_SEGS_FOR_VOL(4); }\
    if (do_trace[5]) { GET_POSITION_FOR_VOL(5); LOAD_SEGS_FOR_VOL(5); }\
    if (do_trace[6]) { GET_POSITION_FOR_VOL(6); LOAD_SEGS_FOR_VOL(6); }\
} while (0)
#elif NUM_VOLUMES == 8
#define LOAD_SEGS_AT_ALPHA do {\
    if (do_trace[0]) { GET_POSITION_FOR_VOL(0); LOAD_SEGS_FOR_VOL(0); }\
    if (do_trace[1]) { GET_POSITION_FOR_VOL(1); LOAD_SEGS_FOR_VOL(1); }\
    if (do_trace[2]) { GET_POSITION_FOR_VOL(2); LOAD_SEGS_FOR_VOL(2); }\
    if (do_trace[3]) { GET_POSITION_FOR_VOL(3); LOAD_SEGS_FOR_VOL(3); }\
    if (do_trace[4]) { GET_POSITION_FOR_VOL(4); LOAD_SEGS_FOR_VOL(4); }\
    if (do_trace[5]) { GET_POSITION_FOR_VOL(5); LOAD_SEGS_FOR_VOL(5); }\
    if (do_trace[6]) { GET_POSITION_FOR_VOL(6); LOAD_SEGS_FOR_VOL(6); }\
    if (do_trace[7]) { GET_POSITION_FOR_VOL(7); LOAD_SEGS_FOR_VOL(7); }\
} while (0)
#elif NUM_VOLUMES == 9
#define LOAD_SEGS_AT_ALPHA do {\
    if (do_trace[0]) { GET_POSITION_FOR_VOL(0); LOAD_SEGS_FOR_VOL(0); }\
    if (do_trace[1]) { GET_POSITION_FOR_VOL(1); LOAD_SEGS_FOR_VOL(1); }\
    if (do_trace[2]) { GET_POSITION_FOR_VOL(2); LOAD_SEGS_FOR_VOL(2); }\
    if (do_trace[3]) { GET_POSITION_FOR_VOL(3); LOAD_SEGS_FOR_VOL(3); }\
    if (do_trace[4]) { GET_POSITION_FOR_VOL(4); LOAD_SEGS_FOR_VOL(4); }\
    if (do_trace[5]) { GET_POSITION_FOR_VOL(5); LOAD_SEGS_FOR_VOL(5); }\
    if (do_trace[6]) { GET_POSITION_FOR_VOL(6); LOAD_SEGS_FOR_VOL(6); }\
    if (do_trace[7]) { GET_POSITION_FOR_VOL(7); LOAD_SEGS_FOR_VOL(7); }\
    if (do_trace[8]) { GET_POSITION_FOR_VOL(8); LOAD_SEGS_FOR_VOL(8); }\
} while (0)
#elif NUM_VOLUMES == 10
#define LOAD_SEGS_AT_ALPHA do {\
    if (do_trace[0]) { GET_POSITION_FOR_VOL(0); LOAD_SEGS_FOR_VOL(0); }\
    if (do_trace[1]) { GET_POSITION_FOR_VOL(1); LOAD_SEGS_FOR_VOL(1); }\
    if (do_trace[2]) { GET_POSITION_FOR_VOL(2); LOAD_SEGS_FOR_VOL(2); }\
    if (do_trace[3]) { GET_POSITION_FOR_VOL(3); LOAD_SEGS_FOR_VOL(3); }\
    if (do_trace[4]) { GET_POSITION_FOR_VOL(4); LOAD_SEGS_FOR_VOL(4); }\
    if (do_trace[5]) { GET_POSITION_FOR_VOL(5); LOAD_SEGS_FOR_VOL(5); }\
    if (do_trace[6]) { GET_POSITION_FOR_VOL(6); LOAD_SEGS_FOR_VOL(6); }\
    if (do_trace[7]) { GET_POSITION_FOR_VOL(7); LOAD_SEGS_FOR_VOL(7); }\
    if (do_trace[8]) { GET_POSITION_FOR_VOL(8); LOAD_SEGS_FOR_VOL(8); }\
    if (do_trace[9]) { GET_POSITION_FOR_VOL(9); LOAD_SEGS_FOR_VOL(9); }\
} while (0)
#else
#define LOAD_SEGS_AT_ALPHA do {\
    fprintf(stderr, "LOAD_SEGS_AT_ALPHA not supported for NUM_VOLUMES outside [1, 10]");\
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
    if (do_trace[0] && (priority[0] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 0); }\
    if (do_trace[1] && (priority[1] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 1); }\
} while (0)
#elif NUM_VOLUMES == 3
#define INTERPOLATE(multiplier) do {\
    if (do_trace[0] && (priority[0] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 0); }\
    if (do_trace[1] && (priority[1] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 1); }\
    if (do_trace[2] && (priority[2] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 2); }\
} while (0)
#elif NUM_VOLUMES == 4
#define INTERPOLATE(multiplier) do {\
    if (do_trace[0] && (priority[0] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 0); }\
    if (do_trace[1] && (priority[1] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 1); }\
    if (do_trace[2] && (priority[2] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 2); }\
    if (do_trace[3] && (priority[3] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 3); }\
} while (0)
#elif NUM_VOLUMES == 5
#define INTERPOLATE(multiplier) do {\
    if (do_trace[0] && (priority[0] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 0); }\
    if (do_trace[1] && (priority[1] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 1); }\
    if (do_trace[2] && (priority[2] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 2); }\
    if (do_trace[3] && (priority[3] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 3); }\
    if (do_trace[4] && (priority[4] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 4); }\
} while (0)
#elif NUM_VOLUMES == 6
#define INTERPOLATE(multiplier) do {\
    if (do_trace[0] && (priority[0] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 0); }\
    if (do_trace[1] && (priority[1] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 1); }\
    if (do_trace[2] && (priority[2] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 2); }\
    if (do_trace[3] && (priority[3] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 3); }\
    if (do_trace[4] && (priority[4] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 4); }\
    if (do_trace[5] && (priority[5] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 5); }\
} while (0)
#elif NUM_VOLUMES == 7
#define INTERPOLATE(multiplier) do {\
    if (do_trace[0] && (priority[0] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 0); }\
    if (do_trace[1] && (priority[1] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 1); }\
    if (do_trace[2] && (priority[2] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 2); }\
    if (do_trace[3] && (priority[3] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 3); }\
    if (do_trace[4] && (priority[4] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 4); }\
    if (do_trace[5] && (priority[5] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 5); }\
    if (do_trace[6] && (priority[6] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 6); }\
} while (0)
#elif NUM_VOLUMES == 8
#define INTERPOLATE(multiplier) do {\
    if (do_trace[0] && (priority[0] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 0); }\
    if (do_trace[1] && (priority[1] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 1); }\
    if (do_trace[2] && (priority[2] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 2); }\
    if (do_trace[3] && (priority[3] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 3); }\
    if (do_trace[4] && (priority[4] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 4); }\
    if (do_trace[5] && (priority[5] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 5); }\
    if (do_trace[6] && (priority[6] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 6); }\
    if (do_trace[7] && (priority[7] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 7); }\
} while (0)
#elif NUM_VOLUMES == 9
#define INTERPOLATE(multiplier) do {\
    if (do_trace[0] && (priority[0] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 0); }\
    if (do_trace[1] && (priority[1] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 1); }\
    if (do_trace[2] && (priority[2] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 2); }\
    if (do_trace[3] && (priority[3] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 3); }\
    if (do_trace[4] && (priority[4] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 4); }\
    if (do_trace[5] && (priority[5] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 5); }\
    if (do_trace[6] && (priority[6] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 6); }\
    if (do_trace[7] && (priority[7] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 7); }\
    if (do_trace[8] && (priority[8] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 8); }\
} while (0)
#elif NUM_VOLUMES == 10
#define INTERPOLATE(multiplier) do {\
    if (do_trace[0] && (priority[0] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 0); }\
    if (do_trace[1] && (priority[1] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 1); }\
    if (do_trace[2] && (priority[2] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 2); }\
    if (do_trace[3] && (priority[3] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 3); }\
    if (do_trace[4] && (priority[4] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 4); }\
    if (do_trace[5] && (priority[5] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 5); }\
    if (do_trace[6] && (priority[6] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 6); }\
    if (do_trace[7] && (priority[7] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 7); }\
    if (do_trace[8] && (priority[8] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 8); }\
    if (do_trace[9] && (priority[9] == curr_priority)) { INTERPOLATE_FOR_VOL(multiplier, 9); }\
} while (0)
#else
#define INTERPOLATE(multiplier) do {\
    fprintf(stderr, "INTERPOLATE not supported for NUM_VOLUMES outside [1, 10]");\
} while (0)
#endif

extern "C" {
    /* "return" variables point to an item in the array, not the beginning of the array */
    __device__ static void calculate_alpha(
        float *minAlpha, float *maxAlpha, int *do_trace, 
        float *globalMinAlpha, float *globalMaxAlpha,
        float rx, float ry, float rz,
        float sx, float sy, float sz,
        float minBoundX, float minBoundY, float minBoundZ,
        float maxBoundX, float maxBoundY, float maxBoundZ
    ) {
        *minAlpha = 0.0f;
        *maxAlpha = INFINITY;
        *do_trace = 1;

        if (0.0f != rx) {
            float reci = 1.0f / rx;
            float alpha0 = (minBoundX - sx) * reci;
            float alpha1 = (maxBoundX - sx) * reci;
            *minAlpha = fmin(alpha0, alpha1);
            *maxAlpha = fmax(alpha0, alpha1);
        } else if (minBoundX > sx || sx > maxBoundX) {
            *do_trace = 0;
        }
    
        if ((*do_trace) && (0.0f != ry)) {
            float reci = 1.0f / ry;
            float alpha0 = (minBoundY - sy) * reci;
            float alpha1 = (maxBoundY - sy) * reci;
            *minAlpha = fmax(*minAlpha, fmin(alpha0, alpha1));
            *maxAlpha = fmin(*maxAlpha, fmax(alpha0, alpha1));
        } else if (minBoundY > sy || sy > maxBoundY) {
            *do_trace = 0;
        }
    
        if ((*do_trace) && (0.0f != rz))  {
            float reci = 1.0f / rz;
            float alpha0 = (minBoundZ - sz) * reci;
            float alpha1 = (maxBoundZ - sz) * reci;
            *minAlpha = fmax(*minAlpha, fmin(alpha0, alpha1));
            *maxAlpha = fmin(*maxAlpha, fmax(alpha0, alpha1));
        } else if (minBoundZ > sz || sz > maxBoundZ) {
            *do_trace = 0;
        }
        *globalMinAlpha = fmin(*minAlpha, *globalMinAlpha);
        *globalMaxAlpha = fmax(*maxAlpha, *globalMaxAlpha);
    }

    __device__ static void calculate_all_alphas(
        float minAlpha[NUM_VOLUMES], float maxAlpha[NUM_VOLUMES], int do_trace[NUM_VOLUMES],
        float *globalMinAlpha, float *globalMaxAlpha, 
        float rx[NUM_VOLUMES], float ry[NUM_VOLUMES], float rz[NUM_VOLUMES],
        float sx[NUM_VOLUMES], float sy[NUM_VOLUMES], float sz[NUM_VOLUMES],
        float gVolumeEdgeMinPointX[NUM_VOLUMES], float gVolumeEdgeMinPointY[NUM_VOLUMES], float gVolumeEdgeMinPointZ[NUM_VOLUMES], 
        float gVolumeEdgeMaxPointX[NUM_VOLUMES], float gVolumeEdgeMaxPointY[NUM_VOLUMES], float gVolumeEdgeMaxPointZ[NUM_VOLUMES]
    ) {
        #if NUM_VOLUMES <= 0
        fprintf(stderr, "calculate_all_alphas not supported for NUM_VOLUMES outside [1, 10]"); return;
        #endif

        #if NUM_VOLUMES > 10
        fprintf(stderr, "calculate_all_alphas not supported for NUM_VOLUMES outside [1, 10]"); return;
        #endif

        int i; 
        #if NUM_VOLUMES > 0
        i = 0;
        calculate_alpha(
            &minAlpha[i], &maxAlpha[i], &do_trace[i],
            globalMinAlpha, globalMaxAlpha,
            rx[i], ry[i], rz[i],
            sx[i], sy[i], sz[i],
            gVolumeEdgeMinPointX[i], gVolumeEdgeMinPointY[i], gVolumeEdgeMinPointZ[i],
            gVolumeEdgeMaxPointX[i], gVolumeEdgeMaxPointY[i], gVolumeEdgeMaxPointZ[i]
        );
        #endif
        #if NUM_VOLUMES > 1
        i = 1;
        calculate_alpha(
            &minAlpha[i], &maxAlpha[i], &do_trace[i],
            globalMinAlpha, globalMaxAlpha,
            rx[i], ry[i], rz[i],
            sx[i], sy[i], sz[i],
            gVolumeEdgeMinPointX[i], gVolumeEdgeMinPointY[i], gVolumeEdgeMinPointZ[i],
            gVolumeEdgeMaxPointX[i], gVolumeEdgeMaxPointY[i], gVolumeEdgeMaxPointZ[i]
        );
        #endif
        #if NUM_VOLUMES > 2
        i = 2;
        calculate_alpha(
            &minAlpha[i], &maxAlpha[i], &do_trace[i],
            globalMinAlpha, globalMaxAlpha,
            rx[i], ry[i], rz[i],
            sx[i], sy[i], sz[i],
            gVolumeEdgeMinPointX[i], gVolumeEdgeMinPointY[i], gVolumeEdgeMinPointZ[i],
            gVolumeEdgeMaxPointX[i], gVolumeEdgeMaxPointY[i], gVolumeEdgeMaxPointZ[i]
        );
        #endif
        #if NUM_VOLUMES > 3
        i = 3;
        calculate_alpha(
            &minAlpha[i], &maxAlpha[i], &do_trace[i],
            globalMinAlpha, globalMaxAlpha,
            rx[i], ry[i], rz[i],
            sx[i], sy[i], sz[i],
            gVolumeEdgeMinPointX[i], gVolumeEdgeMinPointY[i], gVolumeEdgeMinPointZ[i],
            gVolumeEdgeMaxPointX[i], gVolumeEdgeMaxPointY[i], gVolumeEdgeMaxPointZ[i]
        );
        #endif
        #if NUM_VOLUMES > 4
        i = 4;
        calculate_alpha(
            &minAlpha[i], &maxAlpha[i], &do_trace[i],
            globalMinAlpha, globalMaxAlpha,
            rx[i], ry[i], rz[i],
            sx[i], sy[i], sz[i],
            gVolumeEdgeMinPointX[i], gVolumeEdgeMinPointY[i], gVolumeEdgeMinPointZ[i],
            gVolumeEdgeMaxPointX[i], gVolumeEdgeMaxPointY[i], gVolumeEdgeMaxPointZ[i]
        );
        #endif
        #if NUM_VOLUMES > 5
        i = 5;
        calculate_alpha(
            &minAlpha[i], &maxAlpha[i], &do_trace[i],
            globalMinAlpha, globalMaxAlpha,
            rx[i], ry[i], rz[i],
            sx[i], sy[i], sz[i],
            gVolumeEdgeMinPointX[i], gVolumeEdgeMinPointY[i], gVolumeEdgeMinPointZ[i],
            gVolumeEdgeMaxPointX[i], gVolumeEdgeMaxPointY[i], gVolumeEdgeMaxPointZ[i]
        );
        #endif
        #if NUM_VOLUMES > 6
        i = 6;
        calculate_alpha(
            &minAlpha[i], &maxAlpha[i], &do_trace[i],
            globalMinAlpha, globalMaxAlpha,
            rx[i], ry[i], rz[i],
            sx[i], sy[i], sz[i],
            gVolumeEdgeMinPointX[i], gVolumeEdgeMinPointY[i], gVolumeEdgeMinPointZ[i],
            gVolumeEdgeMaxPointX[i], gVolumeEdgeMaxPointY[i], gVolumeEdgeMaxPointZ[i]
        );
        #endif
        #if NUM_VOLUMES > 7
        i = 7;
        calculate_alpha(
            &minAlpha[i], &maxAlpha[i], &do_trace[i],
            globalMinAlpha, globalMaxAlpha,
            rx[i], ry[i], rz[i],
            sx[i], sy[i], sz[i],
            gVolumeEdgeMinPointX[i], gVolumeEdgeMinPointY[i], gVolumeEdgeMinPointZ[i],
            gVolumeEdgeMaxPointX[i], gVolumeEdgeMaxPointY[i], gVolumeEdgeMaxPointZ[i]
        );
        #endif
        #if NUM_VOLUMES > 8
        i = 8;
        calculate_alpha(
            &minAlpha[i], &maxAlpha[i], &do_trace[i],
            globalMinAlpha, globalMaxAlpha,
            rx[i], ry[i], rz[i],
            sx[i], sy[i], sz[i],
            gVolumeEdgeMinPointX[i], gVolumeEdgeMinPointY[i], gVolumeEdgeMinPointZ[i],
            gVolumeEdgeMaxPointX[i], gVolumeEdgeMaxPointY[i], gVolumeEdgeMaxPointZ[i]
        );
        #endif
        #if NUM_VOLUMES > 9
        i = 9;
        calculate_alpha(
            &minAlpha[i], &maxAlpha[i], &do_trace[i],
            globalMinAlpha, globalMaxAlpha,
            rx[i], ry[i], rz[i],
            sx[i], sy[i], sz[i],
            gVolumeEdgeMinPointX[i], gVolumeEdgeMinPointY[i], gVolumeEdgeMinPointZ[i],
            gVolumeEdgeMaxPointX[i], gVolumeEdgeMaxPointY[i], gVolumeEdgeMaxPointZ[i]
        );
        #endif
    }

    /* "return" variables point to an item in the array, not the beginning of the array */
    __device__ static void calculate_ray(
        float *rx, float *ry, float *rz, float *vnf,
        float u, float v, float *rt_kinv_arr, int rt_kinv_offset,
        float voxelSizeX, float voxelSizeY, float voxelSizeZ
    ) {
        *rx = u * rt_kinv_arr[rt_kinv_offset + 0] + v * rt_kinv_arr[rt_kinv_offset + 1] + rt_kinv_arr[rt_kinv_offset + 2];
        *ry = u * rt_kinv_arr[rt_kinv_offset + 3] + v * rt_kinv_arr[rt_kinv_offset + 4] + rt_kinv_arr[rt_kinv_offset + 5];
        *rz = u * rt_kinv_arr[rt_kinv_offset + 6] + v * rt_kinv_arr[rt_kinv_offset + 7] + rt_kinv_arr[rt_kinv_offset + 8];
        /* make the ray a unit vector */
        float normFactor = 1.0f / sqrt(((*rx) * (*rx)) + ((*ry) * (*ry)) + ((*rz) * (*rz)));
        *rx *= normFactor;
        *ry *= normFactor;
        *rz *= normFactor;
        
        float tmp = 0.0f;
        tmp += ((*rx) * voxelSizeX) * ((*rx) * voxelSizeX);
        tmp += ((*ry) * voxelSizeY) * ((*ry) * voxelSizeY);
        tmp += ((*rz) * voxelSizeZ) * ((*rz) * voxelSizeZ);
        *vnf = sqrtf(tmp);
    }

    __device__ static void calculate_all_rays(
        float rx[NUM_VOLUMES], float ry[NUM_VOLUMES], float rz[NUM_VOLUMES], float volume_normalization_factor[NUM_VOLUMES],
        float u, float v, float rt_kinv_arr[9 * NUM_VOLUMES],
        float gVoxelElementSizeX[NUM_VOLUMES], float gVoxelElementSizeY[NUM_VOLUMES], float gVoxelElementSizeZ[NUM_VOLUMES]
    ) {
        #if NUM_VOLUMES <= 0
        fprintf(stderr, "calculate_all_rays not supported for NUM_VOLUMES outside [1, 10]"); return;
        #endif

        #if NUM_VOLUMES > 10
        fprintf(stderr, "calculate_all_rays not supported for NUM_VOLUMES outside [1, 10]"); return;
        #endif

        int i; 
        #if NUM_VOLUMES > 0
        i = 0;
        calculate_ray(
            &rx[i], &ry[i], &rz[i], &volume_normalization_factor[i],
            u, v, rt_kinv_arr, 9 * i,
            gVoxelElementSizeX[i], gVoxelElementSizeY[i], gVoxelElementSizeZ[i]
        );
        #endif
        #if NUM_VOLUMES > 1
        i = 1;
        calculate_ray(
            &rx[i], &ry[i], &rz[i], &volume_normalization_factor[i],
            u, v, rt_kinv_arr, 9 * i,
            gVoxelElementSizeX[i], gVoxelElementSizeY[i], gVoxelElementSizeZ[i]
        );
        #endif
        #if NUM_VOLUMES > 2
        i = 2;
        calculate_ray(
            &rx[i], &ry[i], &rz[i], &volume_normalization_factor[i],
            u, v, rt_kinv_arr, 9 * i,
            gVoxelElementSizeX[i], gVoxelElementSizeY[i], gVoxelElementSizeZ[i]
        );
        #endif
        #if NUM_VOLUMES > 3
        i = 3;
        calculate_ray(
            &rx[i], &ry[i], &rz[i], &volume_normalization_factor[i],
            u, v, rt_kinv_arr, 9 * i,
            gVoxelElementSizeX[i], gVoxelElementSizeY[i], gVoxelElementSizeZ[i]
        );
        #endif
        #if NUM_VOLUMES > 4
        i = 4;
        calculate_ray(
            &rx[i], &ry[i], &rz[i], &volume_normalization_factor[i],
            u, v, rt_kinv_arr, 9 * i,
            gVoxelElementSizeX[i], gVoxelElementSizeY[i], gVoxelElementSizeZ[i]
        );
        #endif
        #if NUM_VOLUMES > 5
        i = 5;
        calculate_ray(
            &rx[i], &ry[i], &rz[i], &volume_normalization_factor[i],
            u, v, rt_kinv_arr, 9 * i,
            gVoxelElementSizeX[i], gVoxelElementSizeY[i], gVoxelElementSizeZ[i]
        );
        #endif
        #if NUM_VOLUMES > 6
        i = 6;
        calculate_ray(
            &rx[i], &ry[i], &rz[i], &volume_normalization_factor[i],
            u, v, rt_kinv_arr, 9 * i,
            gVoxelElementSizeX[i], gVoxelElementSizeY[i], gVoxelElementSizeZ[i]
        );
        #endif
        #if NUM_VOLUMES > 7
        i = 7;
        calculate_ray(
            &rx[i], &ry[i], &rz[i], &volume_normalization_factor[i],
            u, v, rt_kinv_arr, 9 * i,
            gVoxelElementSizeX[i], gVoxelElementSizeY[i], gVoxelElementSizeZ[i]
        );
        #endif
        #if NUM_VOLUMES > 8
        i = 8;
        calculate_ray(
            &rx[i], &ry[i], &rz[i], &volume_normalization_factor[i],
            u, v, rt_kinv_arr, 9 * i,
            gVoxelElementSizeX[i], gVoxelElementSizeY[i], gVoxelElementSizeZ[i]
        );
        #endif
        #if NUM_VOLUMES > 9
        i = 9;
        calculate_ray(
            &rx[i], &ry[i], &rz[i], &volume_normalization_factor[i],
            u, v, rt_kinv_arr, 9 * i,
            gVoxelElementSizeX[i], gVoxelElementSizeY[i], gVoxelElementSizeZ[i]
        );
        #endif
    }

    __device__ static void get_priority_at_alpha(
        float alpha, int *curr_priority, int *n_vols_at_curr_priority,
        float minAlpha[NUM_VOLUMES], float maxAlpha[NUM_VOLUMES], int do_trace[NUM_VOLUMES],
        float seg_at_alpha[NUM_VOLUMES][NUM_MATERIALS], int priority[NUM_VOLUMES]
    ) {
        *curr_priority = NUM_VOLUMES;
        *n_vols_at_curr_priority = 0;
        for (int i = 0; i < NUM_VOLUMES; i++) {
            if (0 == do_trace[i]) { continue; }
            if ((alpha < minAlpha[i]) || (alpha > maxAlpha[i])) { continue; }
            float any_seg = 0.0f;
            for (int m = 0; m < NUM_MATERIALS; m++) {
                any_seg += seg_at_alpha[i][m];
                if (any_seg > 0.0f) { break; }
            }
            if (0.0f == any_seg) { continue; }
    
            if (priority[i] < *curr_priority) {
                *curr_priority = priority[i];
                *n_vols_at_curr_priority = 1;
            } else if (priority[i] == *curr_priority) {
                *n_vols_at_curr_priority += 1;
            }
        }
    }

    __global__  void projectKernel(
        int out_width, // width of the output image
        int out_height, // height of the output image
        float step,
        int priority[NUM_VOLUMES], // volumes with smaller priority-ID have higher priority when determining which volume we are in
        float gVolumeEdgeMinPointX[NUM_VOLUMES], // one value for each of the NUM_VOLUMES volumes
        float gVolumeEdgeMinPointY[NUM_VOLUMES],
        float gVolumeEdgeMinPointZ[NUM_VOLUMES],
        float gVolumeEdgeMaxPointX[NUM_VOLUMES],
        float gVolumeEdgeMaxPointY[NUM_VOLUMES],
        float gVolumeEdgeMaxPointZ[NUM_VOLUMES],
        float gVoxelElementSizeX[NUM_VOLUMES], // one value for each of the NUM_VOLUMES volumes
        float gVoxelElementSizeY[NUM_VOLUMES],
        float gVoxelElementSizeZ[NUM_VOLUMES],
        float sx[NUM_VOLUMES], // x-coordinate of source point for rays in world-space
        float sy[NUM_VOLUMES], // one value for each of the NUM_VOLUMES volumes
        float sz[NUM_VOLUMES],
        float rt_kinv[9 * NUM_VOLUMES], // (NUM_VOLUMES, 3, 3) array giving the image-to-world-ray transform for each volume
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

        // cell-centered sampling point corresponding to pixel index, in index-space.
        float u = (float) udx + 0.5;
        float v = (float) vdx + 0.5;

        // Vector in voxel-space along ray from source-point to pixel at [u,v] on the detector plane.
        float rx[NUM_VOLUMES];
        float ry[NUM_VOLUMES];
        float rz[NUM_VOLUMES];
        float volume_normalization_factor[NUM_VOLUMES];
        calculate_all_rays(
            rx, ry, rz, volume_normalization_factor,
            u, v, rt_kinv, 
            gVoxelElementSizeX, gVoxelElementSizeY, gVoxelElementSizeZ
        );

        // calculate projections
        // Part 1: compute alpha value at entry and exit point of the volume on either side of the ray.
        // minAlpha: the distance from source point to volume entry point of the ray.
        // maxAlpha: the distance from source point to volume exit point of the ray.
        float minAlpha[NUM_VOLUMES];
        float maxAlpha[NUM_VOLUMES];
        int do_trace[NUM_VOLUMES]; // for each volume, whether or not to perform the ray-tracing
        float globalMinAlpha = INFINITY; // the smallest of all the minAlpha's
        float globalMaxAlpha = 0.0f; // the largest of all the maxAlpha's
        calculate_all_alphas(
            minAlpha, maxAlpha, do_trace,
            &globalMinAlpha, &globalMaxAlpha,
            rx, ry, rz,
            sx, sy, sz,
            gVolumeEdgeMinPointX, gVolumeEdgeMinPointY, gVolumeEdgeMinPointZ,
            gVolumeEdgeMaxPointX, gVolumeEdgeMaxPointY, gVolumeEdgeMaxPointZ
        );

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
        int n_vols_at_curr_priority;//B[NUM_MATERIALS]; // how many volumes to consider at the location (for each material)
        float seg_at_alpha[NUM_VOLUMES][NUM_MATERIALS];

        for (alpha = globalMinAlpha; alpha < globalMaxAlpha; alpha += step) {
            LOAD_SEGS_AT_ALPHA; // initializes p{x,y,z}[...] and seg_at_alpha[...][...]
            get_priority_at_alpha(
                alpha, &curr_priority, &n_vols_at_curr_priority,
                minAlpha, maxAlpha, do_trace,
                seg_at_alpha, priority
            );
            if (0 == n_vols_at_curr_priority) {
                // Outside the bounds of all volumes to trace. Assume nominal density of air is 0.0f.
                // Thus, we don't need to add to area_density
                ;
            } else {
                float weight = 1.0f / ((float)n_vols_at_curr_priority);

                // For the entry boundary, multiply by 0.5. That is, for the initial interpolated value,
                // only a half step-size is considered in the computation. For the second-to-last interpolation
                // point, also multiply by 0.5, since there will be a final step at the globalMaxAlpha boundary.
                boundary_factor = ((alpha <= globalMinAlpha) || (alpha + step >= globalMaxAlpha)) ? 0.5f : 1.0f;

                INTERPOLATE(boundary_factor);
            }
        }

        // Scaling by step
        for (int m = 0; m < NUM_MATERIALS; m++) {
            area_density[m] *= step;
        }

        // Last segment of the line
        if (area_density[0] > 0.0f) {
            alpha -= step;
            float lastStepsize = globalMaxAlpha - alpha;
            
            if (0 == n_vols_at_curr_priority) {
                // Outside the bounds of all volumes to trace. Assume nominal density of air is 0.0f.
                // Thus, we don't need to add to area_density
                ;
            } else {
                float weight = 1.0f / ((float)n_vols_at_curr_priority);

                // Scaled last step interpolation (something weird?)
                INTERPOLATE(lastStepsize);
            }
        }

        // Convert to centimeters
        for (int m = 0; m < NUM_MATERIALS; m++) {
            area_density[m] /= 10.0f;
        }

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
    
