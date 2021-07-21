#include <stdio.h>
#include <cubicTex3D.cu>

#include "kernel_vol_seg_data.cu"

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
    printf("NUM_MATERIALS not in [1, 14]");\
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
    printf("LOAD_SEGS_AT_ALPHA not supported for NUM_VOLUMES outside [1, 10]");\
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
    printf("NUM_MATERIALS not in [1, 14]");\
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

#define PI_FLOAT  3.14159265358979323846f
#define FOUR_PI_INV_FLOAT 0.0795774715459476678844f // 1 / (4 \pi), from Wolfram Alpha

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
        printf("calculate_all_alphas not supported for NUM_VOLUMES outside [1, 10]"); return;
        #endif

        #if NUM_VOLUMES > 10
        printf("calculate_all_alphas not supported for NUM_VOLUMES outside [1, 10]"); return;
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
        printf("calculate_all_rays not supported for NUM_VOLUMES outside [1, 10]"); return;
        #endif

        #if NUM_VOLUMES > 10
        printf("calculate_all_rays not supported for NUM_VOLUMES outside [1, 10]"); return;
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
        float step, // step size (TODO: in world)
        int *priority, // volumes with smaller priority-ID have higher priority when determining which volume we are in
        float *gVolumeEdgeMinPointX, // These give a bounding box in world-space around each volume.
        float *gVolumeEdgeMinPointY,
        float *gVolumeEdgeMinPointZ,
        float *gVolumeEdgeMaxPointX,
        float *gVolumeEdgeMaxPointY,
        float *gVolumeEdgeMaxPointZ,
        float *gVoxelElementSizeX, // one value for each of the NUM_VOLUMES volumes
        float *gVoxelElementSizeY,
        float *gVoxelElementSizeZ,
        float sx, // x-coordinate of source point for rays in world-space
        float sy,
        float sz,
        float *rt_kinv, // (3, 3) array giving the ijk_from_index ray transform for each volume (todo: make it a single world_from_index array)
        float *ijk_from_world, // (NUM_VOLUMES, 3, 4) transform giving the transform from world to IJK coordinates for each volume.
        int n_bins, // the number of spectral bins
        float *energies, // 1-D array -- size is the n_bins. Units: [keV]
        float *pdf, // 1-D array -- probability density function over the energies
        float *absorb_coef_table, // flat [n_bins x NUM_MATERIALS] table that represents
                        // the precomputed get_absorption_coef values.
                        // index into the table as: table[bin * NUM_MATERIALS + mat]
        float *intensity, // flat array, with shape (out_height, out_width).
        float *photon_prob, // flat array, with shape (out_height, out_width).
        float *solid_angle, // flat array, with shape (out_height, out_width). Could be NULL pointer
        int offsetW,
        int offsetH)
    {
        // The output image has the following coordinate system, with cell-centered sampling.
        // y is along the fast axis (columns), x along the slow (rows).
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
        // int debug = (udx == 973) && (vdx == 598); // larger image size
        int debug = (udx == 243) && (vdx == 149); // 4x4 binning

        // if the current point is outside the output image, no computation needed
        if (udx >= out_width || vdx >= out_height)
            return;

        // cell-centered sampling point corresponding to pixel index, in index-space.
        float u = (float) udx + 0.5;
        float v = (float) vdx + 0.5;

        // Vector in world-space along ray from source-point to pixel at [u,v] on the detector plane.
        float rx = u * rt_kinv[0] + v * rt_kinv[1] + rt_kinv[2];
        float ry = u * rt_kinv[3] + v * rt_kinv[4] + rt_kinv[5];
        float rz = u * rt_kinv[6] + v * rt_kinv[7] + rt_kinv[8];

        /* make the ray a unit vector */
        float ray_norm = sqrt(rx * rx + ry * ry + rz * rz);
        rx /= ray_norm;
        ry /= ray_norm;
        rz /= ray_norm;

        // calculate projections
        // Part 1: compute alpha value at entry and exit point of all volumes on either side of the ray, in world-space.
        // minAlpha: the distance from source point to all-volumes entry point of the ray, in world-space.
        // maxAlpha: the distance from source point to all-volumes exit point of the ray.
        float minAlpha = 0; // the furthest along the ray we want to consider is the start point.
        float maxAlpha = ray_norm; // closest point to consider is at the detector
        int do_trace[NUM_VOLUMES]; // for each volume, whether or not to perform the ray-tracing
        int do_return = 1;

        for (int i = 0; i < NUM_VOLUMES; i++) {
            do_trace[i] = 1;

            if (0.0f != rx) {
                float reci = 1.0f / rx;
                float alpha0 = (gVolumeEdgeMinPointX[i] - sx) * reci;
                float alpha1 = (gVolumeEdgeMaxPointX[i] - sx) * reci;
                minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
                maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
            } else if (gVolumeEdgeMinPointX[i] > sx || sx > gVolumeEdgeMaxPointX[i]) {
                do_trace[i] = 0;
                continue;
            }

            if (0.0f != ry) {
                float reci = 1.0f / ry;
                float alpha0 = (gVolumeEdgeMinPointY[i] - sy) * reci;
                float alpha1 = (gVolumeEdgeMaxPointY[i] - sy) * reci;
                minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
                maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
            } else if (gVolumeEdgeMinPointY[i] > sy || sy > gVolumeEdgeMaxPointY[i]) {
                do_trace[i] = 0;
                continue;
            }

            if (0.0f != rz) {
                float reci = 1.0f / rz;
                float alpha0 = (gVolumeEdgeMinPointZ[i] - sz) * reci;
                float alpha1 = (gVolumeEdgeMaxPointZ[i] - sz) * reci;
                minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
                maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
            } else if (gVolumeEdgeMinPointZ[i] > sz || sz > gVolumeEdgeMaxPointZ[i]) {
                do_trace[i] = 0;
                continue;
            }

            do_return = 0;
        }

        if (debug) printf("global min, max alphas: %f, %f", minAlpha, maxAlpha);

        // Means none of the volumes have do_trace = 1.
        if (do_return) return;

        printf("CRITICAL: this kernel not finished");

        // TODO: finish using world-space points.

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

        if (debug) {
            printf("start trace\n"); // This is the one that seems to take a half a second.
        }
        int num_steps = 0;
        for (alpha = globalMinAlpha; alpha < globalMaxAlpha; alpha += step, num_steps++) {
            LOAD_SEGS_AT_ALPHA; // initializes p{x,y,z}[...] and seg_at_alpha[...][...]
            // if (debug) printf("  loaded segs\n"); // This is the one that seems to take a half a second.
            get_priority_at_alpha(
                alpha, &curr_priority, &n_vols_at_curr_priority,
                minAlpha, maxAlpha, do_trace,
                seg_at_alpha, priority
            );
            // if (debug) printf("  got priority at alpha, num vols\n"); // This is the one that seems to take a half a second.
            if (0 == n_vols_at_curr_priority) {
                // Outside the bounds of all volumes to trace. Assume nominal density of air is 0.0f.
                // Thus, we don't need to add to area_density
                ;
            } else {
                float weight = 1.0f / ((float) n_vols_at_curr_priority);

                // For the entry boundary, multiply by 0.5. That is, for the initial interpolated value,
                // only a half step-size is considered in the computation. For the second-to-last interpolation
                // point, also multiply by 0.5, since there will be a final step at the globalMaxAlpha boundary.
                boundary_factor = ((alpha <= globalMinAlpha) || (alpha + step >= globalMaxAlpha)) ? 0.5f : 1.0f;

                INTERPOLATE(boundary_factor);
            }

            // if (debug) printf("  interpolated\n"); // This is the one that seems to take a half a second.
        }
       if (debug) printf("finished trace, num_steps: %d\n", num_steps);

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
        if (debug) {
            printf("attenuation\n");
        }
        for (int bin = 0; bin < n_bins; bin++) {
            float beer_lambert_exp = 0.0f;
            for (int m = 0; m < NUM_MATERIALS; m++) {
                beer_lambert_exp += area_density[m] * absorb_coef_table[bin * NUM_MATERIALS + m];
            }
            float photon_prob_tmp = expf(-1.f * beer_lambert_exp) * pdf[bin]; // dimensionless value

            photon_prob[img_dx] += photon_prob_tmp;
            intensity[img_dx] += energies[bin] * photon_prob_tmp; // units: [keV] per unit photon to hit the pixel
        }
        if (debug) {
            printf("done with attenuation\n");
        }
        if (NULL != solid_angle) {
            /**
            * SOLID ANGLE CALCULATION
            *
            * Let the pixel's four corners be c0, c1, c2, c3.  Split the pixel into two right
            * triangles.  These triangles each form a tetrahedron with the X-ray source S.  We
            * can then use a solid-angle-of-tetrahedron formula.
            * 
            * From Wikipedia:
            *      Let OABC be the vertices of a tetrahedron with an origin at O subtended by
            * the triangular face ABC where \vec{a}, \vec{b}, \vec{c} are the vectors \vec{SA},
            * \vec{SB}, \vec{SC} respectively.  Then,
            *
            * tan(\Omega / 2) = NUMERATOR / DENOMINATOR, with
            *
            * NUMERATOR = \vec{a} \cdot (\vec{b} \times \vec{c})
            * DENOMINATOR = abc + (\vec{a} \cdot \vec{b}) c + (\vec{a} \cdot \vec{c}) b + (\vec{b} \cdot \vec{c}) a
            * 
            * where a,b,c are the magnitudes of their respective vectors.
            *
            * There are two potential pitfalls with the above formula.
            * 1. The NUMERATOR (a scalar triple product) can be negative if \vec{a}, \vec{b}, 
            *  \vec{c} have the wrong winding.  Since no other portion of the formula depends
            *  on the winding, computing the absolute value of the scalar triple product is 
            *  sufficient.
            * 2. If the NUMERATOR is positive but the DENOMINATOR is negative, the formula 
            *  returns a negative value that must be increased by \pi.
            */

            /*
            * PIXEL DIAGRAM
            *
            * corner0 __ corner1
            *        |__|
            * corner3    corner2
            */
            float cx[4]; // source-to-corner vector x-values
            float cy[4]; // source-to-corner vector y-values
            float cz[4]; // source-to-corner vector z-values
            float cmag[4]; // magnitude of source-to-corner vector

            float cu_offset[4] = {0.f, 1.f, 1.f, 0.f};
            float cv_offset[4] = {0.f, 0.f, 1.f, 1.f};
            if (debug) {
                printf("solid angle\n");
            }
            for (int i = 0; i < 4; i++) {
                float cu = udx + cu_offset[i];
                float cv = vdx + cv_offset[i];

                cx[i] = cu * rt_kinv[0] + cv * rt_kinv[1] + rt_kinv[2];
                cy[i] = cu * rt_kinv[3] + cv * rt_kinv[4] + rt_kinv[5];
                cz[i] = cu * rt_kinv[6] + cv * rt_kinv[7] + rt_kinv[8];
                
                cmag[i] = (cx[i] * cx[i]) + (cy[i] * cy[i]) + (cz[i] * cz[i]);
                cmag[i] = sqrtf(cmag[i]);
            }
    
            /*
            * The cross- and dot-products needed for the [c0, c1, c2] triangle are:
            *
            * - absolute value of triple product of c0,c1,c2 = c1 \cdot (c0 \times c2)
            *      Since the magnitude of the triple product is invariant under reorderings
            *      of the three vectors, we choose to cross-product c0,c2 so we can reuse
            *      that result
            * - dot product of c0, c1
            * - dot product of c0, c2
            * - dot product of c1, c2
            * 
            * The products needed for the [c0, c2, c3] triangle are:
            *
            * - absolute value of triple product of c0,c2,c3 = c3 \cdot (c0 \times c2)
            *      Since the magnitude of the triple product is invariant under reorderings
            *      of the three vectors, we choose to cross-product c0,c2 so we can reuse
            *      that result
            * - dot product of c0, c2
            * - dot product of c0, c3
            * - dot product of c2, c3
            *
            * Thus, the cross- and dot-products to compute are:
            *  - c0 \times c2
            *  - c0 \dot c1
            *  - c0 \dot c2
            *  - c0 \dot c3
            *  - c1 \dot c2
            *  - c2 \dot c3
            */
            float c0_cross_c2_x = (cy[0] * cz[2]) - (cz[0] * cy[2]);
            float c0_cross_c2_y = (cz[0] * cx[2]) - (cx[0] * cz[2]);
            float c0_cross_c2_z = (cx[0] * cy[2]) - (cy[0] * cx[2]);

            float c0_dot_c1 = (cx[0] * cx[1]) + (cy[0] * cy[1]) + (cz[0] * cz[1]);
            float c0_dot_c2 = (cx[0] * cx[2]) + (cy[0] * cy[2]) + (cz[0] * cz[2]);
            float c0_dot_c3 = (cx[0] * cx[3]) + (cy[0] * cy[3]) + (cz[0] * cz[3]);
            float c1_dot_c2 = (cx[1] * cx[2]) + (cy[1] * cy[2]) + (cz[1] * cz[2]);
            float c2_dot_c3 = (cx[2] * cx[3]) + (cy[2] * cy[3]) + (cz[2] * cz[3]);

            float numer_012 = fabs((cx[1] * c0_cross_c2_x) + (cy[1] * c0_cross_c2_y) + (cz[1] * c0_cross_c2_z));
            float numer_023 = fabs((cx[3] * c0_cross_c2_x) + (cy[3] * c0_cross_c2_y) + (cz[3] * c0_cross_c2_z));

            float denom_012 = (cmag[0] * cmag[1] * cmag[2]) + (c0_dot_c1 * cmag[2]) + (c0_dot_c2 * cmag[1]) + (c1_dot_c2 * cmag[0]);
            float denom_023 = (cmag[0] * cmag[2] * cmag[3]) + (c0_dot_c2 * cmag[3]) + (c0_dot_c3 * cmag[2]) + (c2_dot_c3 * cmag[0]);

            float solid_angle_012 = 2.f * atan2(numer_012, denom_012);
            if (solid_angle_012 < 0.0f) {
                solid_angle_012 += PI_FLOAT;
            }
            float solid_angle_023 = 2.f * atan2(numer_023, denom_023);
            if (solid_angle_023 < 0.0f) {
                solid_angle_023 += PI_FLOAT;
            }

            solid_angle[img_dx] = solid_angle_012 + solid_angle_023;
        }

        if (debug) {
            printf("done with kernel thread\n");
        }
        return;
    }

    /*** KERNEL RESAMPLING FUNCTION ***/
    /**
     * It's placed here so that it can properly access the CUDA textures of the volumes and segmentations
     */

    #if NUM_MATERIALS == 1
    #define RESAMPLE_TEXTURES(vol_id) do {\
        density_sample[vol_id] = tex3D(VOLUME(vol_id), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][0] = cubicTex3D(SEG(vol_id, 0), inp_x, inp_y, inp_z);\
    } while (0)
    #elif NUM_MATERIALS == 2
    #define RESAMPLE_TEXTURES(vol_id) do {\
        density_sample[vol_id] = tex3D(VOLUME(vol_id), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][0] = cubicTex3D(SEG(vol_id, 0), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][1] = cubicTex3D(SEG(vol_id, 1), inp_x, inp_y, inp_z);\
    } while (0)
    #elif NUM_MATERIALS == 3
    #define RESAMPLE_TEXTURES(vol_id) do {\
        density_sample[vol_id] = tex3D(VOLUME(vol_id), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][0] = cubicTex3D(SEG(vol_id, 0), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][1] = cubicTex3D(SEG(vol_id, 1), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][2] = cubicTex3D(SEG(vol_id, 2), inp_x, inp_y, inp_z);\
    } while (0)
    #elif NUM_MATERIALS == 4
    #define RESAMPLE_TEXTURES(vol_id) do {\
        density_sample[vol_id] = tex3D(VOLUME(vol_id), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][0] = cubicTex3D(SEG(vol_id, 0), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][1] = cubicTex3D(SEG(vol_id, 1), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][2] = cubicTex3D(SEG(vol_id, 2), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][3] = cubicTex3D(SEG(vol_id, 3), inp_x, inp_y, inp_z);\
    } while (0)
    #elif NUM_MATERIALS == 5
    #define RESAMPLE_TEXTURES(vol_id) do {\
        density_sample[vol_id] = tex3D(VOLUME(vol_id), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][0] = cubicTex3D(SEG(vol_id, 0), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][1] = cubicTex3D(SEG(vol_id, 1), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][2] = cubicTex3D(SEG(vol_id, 2), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][3] = cubicTex3D(SEG(vol_id, 3), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][4] = cubicTex3D(SEG(vol_id, 4), inp_x, inp_y, inp_z);\
    } while (0)
    #elif NUM_MATERIALS == 6
    #define RESAMPLE_TEXTURES(vol_id) do {\
        density_sample[vol_id] = tex3D(VOLUME(vol_id), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][0] = cubicTex3D(SEG(vol_id, 0), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][1] = cubicTex3D(SEG(vol_id, 1), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][2] = cubicTex3D(SEG(vol_id, 2), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][3] = cubicTex3D(SEG(vol_id, 3), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][4] = cubicTex3D(SEG(vol_id, 4), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][5] = cubicTex3D(SEG(vol_id, 5), inp_x, inp_y, inp_z);\
    } while (0)
    #elif NUM_MATERIALS == 7
    #define RESAMPLE_TEXTURES(vol_id) do {\
        density_sample[vol_id] = tex3D(VOLUME(vol_id), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][0] = cubicTex3D(SEG(vol_id, 0), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][1] = cubicTex3D(SEG(vol_id, 1), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][2] = cubicTex3D(SEG(vol_id, 2), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][3] = cubicTex3D(SEG(vol_id, 3), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][4] = cubicTex3D(SEG(vol_id, 4), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][5] = cubicTex3D(SEG(vol_id, 5), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][6] = cubicTex3D(SEG(vol_id, 6), inp_x, inp_y, inp_z);\
    } while (0)
    #elif NUM_MATERIALS == 8
    #define RESAMPLE_TEXTURES(vol_id) do {\
        density_sample[vol_id] = tex3D(VOLUME(vol_id), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][0] = cubicTex3D(SEG(vol_id, 0), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][1] = cubicTex3D(SEG(vol_id, 1), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][2] = cubicTex3D(SEG(vol_id, 2), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][3] = cubicTex3D(SEG(vol_id, 3), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][4] = cubicTex3D(SEG(vol_id, 4), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][5] = cubicTex3D(SEG(vol_id, 5), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][6] = cubicTex3D(SEG(vol_id, 6), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][7] = cubicTex3D(SEG(vol_id, 7), inp_x, inp_y, inp_z);\
    } while (0)
    #elif NUM_MATERIALS == 9
    #define RESAMPLE_TEXTURES(vol_id) do {\
        density_sample[vol_id] = tex3D(VOLUME(vol_id), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][0] = cubicTex3D(SEG(vol_id, 0), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][1] = cubicTex3D(SEG(vol_id, 1), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][2] = cubicTex3D(SEG(vol_id, 2), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][3] = cubicTex3D(SEG(vol_id, 3), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][4] = cubicTex3D(SEG(vol_id, 4), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][5] = cubicTex3D(SEG(vol_id, 5), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][6] = cubicTex3D(SEG(vol_id, 6), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][7] = cubicTex3D(SEG(vol_id, 7), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][8] = cubicTex3D(SEG(vol_id, 8), inp_x, inp_y, inp_z);\
    } while (0)
    #elif NUM_MATERIALS == 10
    #define RESAMPLE_TEXTURES(vol_id) do {\
        density_sample[vol_id] = tex3D(VOLUME(vol_id), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][0] = cubicTex3D(SEG(vol_id, 0), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][1] = cubicTex3D(SEG(vol_id, 1), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][2] = cubicTex3D(SEG(vol_id, 2), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][3] = cubicTex3D(SEG(vol_id, 3), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][4] = cubicTex3D(SEG(vol_id, 4), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][5] = cubicTex3D(SEG(vol_id, 5), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][6] = cubicTex3D(SEG(vol_id, 6), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][7] = cubicTex3D(SEG(vol_id, 7), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][8] = cubicTex3D(SEG(vol_id, 8), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][9] = cubicTex3D(SEG(vol_id, 9), inp_x, inp_y, inp_z);\
    } while (0)
    #elif NUM_MATERIALS == 11
    #define RESAMPLE_TEXTURES(vol_id) do {\
        density_sample[vol_id] = tex3D(VOLUME(vol_id), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][0] = cubicTex3D(SEG(vol_id, 0), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][1] = cubicTex3D(SEG(vol_id, 1), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][2] = cubicTex3D(SEG(vol_id, 2), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][3] = cubicTex3D(SEG(vol_id, 3), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][4] = cubicTex3D(SEG(vol_id, 4), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][5] = cubicTex3D(SEG(vol_id, 5), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][6] = cubicTex3D(SEG(vol_id, 6), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][7] = cubicTex3D(SEG(vol_id, 7), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][8] = cubicTex3D(SEG(vol_id, 8), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][9] = cubicTex3D(SEG(vol_id, 9), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][10] = cubicTex3D(SEG(vol_id, 10), inp_x, inp_y, inp_z);\
    } while (0)
    #elif NUM_MATERIALS == 12
    #define RESAMPLE_TEXTURES(vol_id) do {\
        density_sample[vol_id] = tex3D(VOLUME(vol_id), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][0] = cubicTex3D(SEG(vol_id, 0), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][1] = cubicTex3D(SEG(vol_id, 1), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][2] = cubicTex3D(SEG(vol_id, 2), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][3] = cubicTex3D(SEG(vol_id, 3), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][4] = cubicTex3D(SEG(vol_id, 4), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][5] = cubicTex3D(SEG(vol_id, 5), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][6] = cubicTex3D(SEG(vol_id, 6), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][7] = cubicTex3D(SEG(vol_id, 7), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][8] = cubicTex3D(SEG(vol_id, 8), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][9] = cubicTex3D(SEG(vol_id, 9), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][10] = cubicTex3D(SEG(vol_id, 10), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][11] = cubicTex3D(SEG(vol_id, 11), inp_x, inp_y, inp_z);\
    } while (0)
    #elif NUM_MATERIALS == 13
    #define RESAMPLE_TEXTURES(vol_id) do {\
        density_sample[vol_id] = tex3D(VOLUME(vol_id), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][0] = cubicTex3D(SEG(vol_id, 0), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][1] = cubicTex3D(SEG(vol_id, 1), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][2] = cubicTex3D(SEG(vol_id, 2), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][3] = cubicTex3D(SEG(vol_id, 3), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][4] = cubicTex3D(SEG(vol_id, 4), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][5] = cubicTex3D(SEG(vol_id, 5), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][6] = cubicTex3D(SEG(vol_id, 6), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][7] = cubicTex3D(SEG(vol_id, 7), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][8] = cubicTex3D(SEG(vol_id, 8), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][9] = cubicTex3D(SEG(vol_id, 9), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][10] = cubicTex3D(SEG(vol_id, 10), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][11] = cubicTex3D(SEG(vol_id, 11), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][12] = cubicTex3D(SEG(vol_id, 12), inp_x, inp_y, inp_z);\
    } while (0)
    #elif NUM_MATERIALS == 14
    #define RESAMPLE_TEXTURES(vol_id) do {\
        density_sample[vol_id] = tex3D(VOLUME(vol_id), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][0] = cubicTex3D(SEG(vol_id, 0), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][1] = cubicTex3D(SEG(vol_id, 1), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][2] = cubicTex3D(SEG(vol_id, 2), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][3] = cubicTex3D(SEG(vol_id, 3), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][4] = cubicTex3D(SEG(vol_id, 4), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][5] = cubicTex3D(SEG(vol_id, 5), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][6] = cubicTex3D(SEG(vol_id, 6), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][7] = cubicTex3D(SEG(vol_id, 7), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][8] = cubicTex3D(SEG(vol_id, 8), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][9] = cubicTex3D(SEG(vol_id, 9), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][10] = cubicTex3D(SEG(vol_id, 10), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][11] = cubicTex3D(SEG(vol_id, 11), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][12] = cubicTex3D(SEG(vol_id, 12), inp_x, inp_y, inp_z);\
        mat_sample[vol_id][13] = cubicTex3D(SEG(vol_id, 13), inp_x, inp_y, inp_z);\
    } while (0)
    #else /////////////////
    #define RESAMPLE_TEXTURES(vol_id) do {\
        printf("NUM_MATERIALS not in [1, 14]");\
    } while (0)
    #endif

    __global__ void resample_megavolume(
        int *inp_priority,
        int *inp_voxelBoundX, // number of voxels in x direction for each volume
        int *inp_voxelBoundY,
        int *inp_voxelBoundZ,
        float *inp_ijk_from_world, // ijk_from_world transforms for input volumes TODO: is each transform 3x3?
        float megaMinX, // bounding box for output megavolume, in world coordinates
        float megaMinY,
        float megaMinZ,
        float megaMaxX,
        float megaMaxY,
        float megaMaxZ,
        float megaVoxelSizeX, // voxel size for output megavolume, in world coordinates
        float megaVoxelSizeY,
        float megaVoxelSizeZ,
        int mega_x_len, // the (exclusive, upper) array index bound of the megavolume
        int mega_y_len,
        int mega_z_len,
        float *output_density, // volume-sized array
        char *output_mat_id, // volume-sized array to hold the material IDs of the voxels,
        int offsetX,
        int offsetY,
        int offsetZ
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

        printf("SCATTER resample\n");

        int x_low = threadIdx.x + (blockIdx.x + offsetX) * blockDim.x; // the x-index of the lowest voxel
        int y_low = threadIdx.y + (blockIdx.y + offsetY) * blockDim.y;
        int z_low = threadIdx.z + (blockIdx.z + offsetZ) * blockDim.z;

        int x_high = min(x_low + blockDim.x, mega_x_len);
        int y_high = min(y_low + blockDim.y, mega_y_len);
        int z_high = min(z_low + blockDim.z, mega_z_len);

        if ((x_low == 0) && (y_low == 0) && (z_low == 0) && (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
            printf("blockDim: {%d, %d, %d}\n", blockDim.x, blockDim.y, blockDim.z);
        }
        
        for (int x_ind = x_low; x_ind < x_high; x_ind++) {
            for (int y_ind = y_low; y_ind < y_high; y_ind++) {
                for (int z_ind = z_low; z_ind < z_high; z_ind++) {
                    float x = megaMinX + (0.5f + (float)x_ind) * megaVoxelSizeX;
                    float y = megaMinY + (0.5f + (float)y_ind) * megaVoxelSizeY;
                    float z = megaMinZ + (0.5f + (float)z_ind) * megaVoxelSizeZ;
                    // for each volume, check whether we are inside its bounds
                    int curr_priority = NUM_VOLUMES;

                    for (int i = 0; i < NUM_VOLUMES; i++) {
                        density_sample[i] = -1.0f; // "reset" this volume's sample

                        int offset = 3 * 4 * i; // TODO: do the matrix multiplication proper
                        float inp_x = (inp_ijk_from_world[offset + 0] * x) + (inp_ijk_from_world[offset + 1] * y) + (inp_ijk_from_world[offset + 2] * z);
                        if ((inp_x < 0.0) || (inp_x >= inp_voxelBoundX[i])) continue; // TODO: make sure this behavior agrees with the behavior of ijk_from_world transforms

                        float inp_y = (inp_ijk_from_world[offset + 3] * x) + (inp_ijk_from_world[offset + 4] * y) + (inp_ijk_from_world[offset + 5] * z);
                        if ((inp_y < 0.0) || (inp_y >= inp_voxelBoundY[i])) continue;

                        float inp_z = (inp_ijk_from_world[offset + 6] * x) + (inp_ijk_from_world[offset + 7] * y) + (inp_ijk_from_world[offset + 8] * z);
                        if ((inp_z < 0.0) || (inp_z >= inp_voxelBoundZ[i])) continue;

                        if (inp_priority[i] < curr_priority) curr_priority = inp_priority[i];
                        else if (inp_priority[i] > curr_priority) continue;

                        // mjudish understands that this is ugly, but it compiles 
                        if      (0 == i) { RESAMPLE_TEXTURES(0); }
                        #if NUM_VOLUMES > 1
                        else if (1 == i) { RESAMPLE_TEXTURES(1); }
                        #endif
                        #if NUM_VOLUMES > 2
                        else if (2 == i) { RESAMPLE_TEXTURES(2); }
                        #endif
                        #if NUM_VOLUMES > 3
                        else if (3 == i) { RESAMPLE_TEXTURES(3); }
                        #endif
                        #if NUM_VOLUMES > 4
                        else if (4 == i) { RESAMPLE_TEXTURES(4); }
                        #endif
                        #if NUM_VOLUMES > 5
                        else if (5 == i) { RESAMPLE_TEXTURES(5); }
                        #endif
                        #if NUM_VOLUMES > 6
                        else if (6 == i) { RESAMPLE_TEXTURES(6); }
                        #endif
                        #if NUM_VOLUMES > 7
                        else if (7 == i) { RESAMPLE_TEXTURES(7); }
                        #endif
                        #if NUM_VOLUMES > 8
                        else if (8 == i) { RESAMPLE_TEXTURES(8); }
                        #endif
                        #if NUM_VOLUMES > 9
                        else if (9 == i) { RESAMPLE_TEXTURES(9); }
                        #endif
                        // Maximum supported value of NUM_VOLUMES is 10
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
    
