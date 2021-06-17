/*
 * This file contains the declarations of the CUDA textures for:
 *  - NUM_VOLUMES CT volumes
 *  - (NUM_MATERIALS * NUM_VOLUMES) segmentation channels
 */

#define SEG_PASTER(vol_id, mat_id) seg_ ## vol_id ## _ ## mat_id
#define SEG(vol_id, mat_id) SEG_PASTER(vol_id, mat_id)
#define VOL_PASTER(vol_id) volume_ ## vol_id
#define VOLUME(vol_id) VOL_PASTER(vol_id)

#ifndef NUM_MATERIALS
#define NUM_MATERIALS 14
#endif

#ifndef NUM_VOLUMES
#define NUM_VOLUMES 1
#endif

/*** Handle one volume ***/
#if NUM_VOLUMES > 0
#define CURR_VOL_ID 0
// the CT volume
texture<float, 3, cudaReadModeElementType> VOLUME(CURR_VOL_ID);

// channel of the materials array, same size as the volume.
#if NUM_MATERIALS > 0
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 0);
#endif
#if NUM_MATERIALS > 1
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 1);
#endif
#if NUM_MATERIALS > 2
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 2);
#endif
#if NUM_MATERIALS > 3
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 3);
#endif
#if NUM_MATERIALS > 4
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 4);
#endif
#if NUM_MATERIALS > 5
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 5);
#endif
#if NUM_MATERIALS > 6
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 6);
#endif
#if NUM_MATERIALS > 7
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 7);
#endif
#if NUM_MATERIALS > 8
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 8);
#endif
#if NUM_MATERIALS > 9
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 9);
#endif
#if NUM_MATERIALS > 10
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 10);
#endif
#if NUM_MATERIALS > 11
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 11);
#endif
#if NUM_MATERIALS > 12
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 12);
#endif
#if NUM_MATERIALS > 13
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 13);
#endif

#undef CURR_VOL_ID
#endif


/*** Handle two volumes ***/
#if NUM_VOLUMES > 1
#define CURR_VOL_ID 1
// the CT volume
texture<float, 3, cudaReadModeElementType> VOLUME(CURR_VOL_ID);

// channel of the materials array, same size as the volume.
#if NUM_MATERIALS > 0
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 0);
#endif
#if NUM_MATERIALS > 1
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 1);
#endif
#if NUM_MATERIALS > 2
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 2);
#endif
#if NUM_MATERIALS > 3
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 3);
#endif
#if NUM_MATERIALS > 4
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 4);
#endif
#if NUM_MATERIALS > 5
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 5);
#endif
#if NUM_MATERIALS > 6
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 6);
#endif
#if NUM_MATERIALS > 7
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 7);
#endif
#if NUM_MATERIALS > 8
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 8);
#endif
#if NUM_MATERIALS > 9
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 9);
#endif
#if NUM_MATERIALS > 10
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 10);
#endif
#if NUM_MATERIALS > 11
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 11);
#endif
#if NUM_MATERIALS > 12
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 12);
#endif
#if NUM_MATERIALS > 13
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 13);
#endif

#undef CURR_VOL_ID
#endif


/*** Handle three volumes ***/
#if NUM_VOLUMES > 2
#define CURR_VOL_ID 2
// the CT volume
texture<float, 3, cudaReadModeElementType> VOLUME(CURR_VOL_ID);

// channel of the materials array, same size as the volume.
#if NUM_MATERIALS > 0
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 0);
#endif
#if NUM_MATERIALS > 1
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 1);
#endif
#if NUM_MATERIALS > 2
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 2);
#endif
#if NUM_MATERIALS > 3
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 3);
#endif
#if NUM_MATERIALS > 4
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 4);
#endif
#if NUM_MATERIALS > 5
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 5);
#endif
#if NUM_MATERIALS > 6
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 6);
#endif
#if NUM_MATERIALS > 7
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 7);
#endif
#if NUM_MATERIALS > 8
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 8);
#endif
#if NUM_MATERIALS > 9
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 9);
#endif
#if NUM_MATERIALS > 10
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 10);
#endif
#if NUM_MATERIALS > 11
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 11);
#endif
#if NUM_MATERIALS > 12
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 12);
#endif
#if NUM_MATERIALS > 13
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 13);
#endif

#undef CURR_VOL_ID
#endif


/*** Handle four volumes ***/
#if NUM_VOLUMES > 3
#define CURR_VOL_ID 3
// the CT volume
texture<float, 3, cudaReadModeElementType> VOLUME(CURR_VOL_ID);

// channel of the materials array, same size as the volume.
#if NUM_MATERIALS > 0
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 0);
#endif
#if NUM_MATERIALS > 1
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 1);
#endif
#if NUM_MATERIALS > 2
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 2);
#endif
#if NUM_MATERIALS > 3
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 3);
#endif
#if NUM_MATERIALS > 4
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 4);
#endif
#if NUM_MATERIALS > 5
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 5);
#endif
#if NUM_MATERIALS > 6
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 6);
#endif
#if NUM_MATERIALS > 7
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 7);
#endif
#if NUM_MATERIALS > 8
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 8);
#endif
#if NUM_MATERIALS > 9
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 9);
#endif
#if NUM_MATERIALS > 10
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 10);
#endif
#if NUM_MATERIALS > 11
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 11);
#endif
#if NUM_MATERIALS > 12
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 12);
#endif
#if NUM_MATERIALS > 13
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 13);
#endif

#undef CURR_VOL_ID
#endif


/*** Handle five volumes ***/
#if NUM_VOLUMES > 4
#define CURR_VOL_ID 4
// the CT volume
texture<float, 3, cudaReadModeElementType> VOLUME(CURR_VOL_ID);

// channel of the materials array, same size as the volume.
#if NUM_MATERIALS > 0
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 0);
#endif
#if NUM_MATERIALS > 1
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 1);
#endif
#if NUM_MATERIALS > 2
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 2);
#endif
#if NUM_MATERIALS > 3
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 3);
#endif
#if NUM_MATERIALS > 4
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 4);
#endif
#if NUM_MATERIALS > 5
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 5);
#endif
#if NUM_MATERIALS > 6
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 6);
#endif
#if NUM_MATERIALS > 7
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 7);
#endif
#if NUM_MATERIALS > 8
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 8);
#endif
#if NUM_MATERIALS > 9
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 9);
#endif
#if NUM_MATERIALS > 10
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 10);
#endif
#if NUM_MATERIALS > 11
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 11);
#endif
#if NUM_MATERIALS > 12
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 12);
#endif
#if NUM_MATERIALS > 13
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 13);
#endif

#undef CURR_VOL_ID
#endif


/*** Handle six volumes ***/
#if NUM_VOLUMES > 5
#define CURR_VOL_ID 5
// the CT volume
texture<float, 3, cudaReadModeElementType> VOLUME(CURR_VOL_ID);

// channel of the materials array, same size as the volume.
#if NUM_MATERIALS > 0
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 0);
#endif
#if NUM_MATERIALS > 1
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 1);
#endif
#if NUM_MATERIALS > 2
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 2);
#endif
#if NUM_MATERIALS > 3
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 3);
#endif
#if NUM_MATERIALS > 4
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 4);
#endif
#if NUM_MATERIALS > 5
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 5);
#endif
#if NUM_MATERIALS > 6
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 6);
#endif
#if NUM_MATERIALS > 7
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 7);
#endif
#if NUM_MATERIALS > 8
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 8);
#endif
#if NUM_MATERIALS > 9
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 9);
#endif
#if NUM_MATERIALS > 10
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 10);
#endif
#if NUM_MATERIALS > 11
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 11);
#endif
#if NUM_MATERIALS > 12
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 12);
#endif
#if NUM_MATERIALS > 13
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 13);
#endif

#undef CURR_VOL_ID
#endif


/*** Handle seven volumes ***/
#if NUM_VOLUMES > 6
#define CURR_VOL_ID 6
// the CT volume
texture<float, 3, cudaReadModeElementType> VOLUME(CURR_VOL_ID);

// channel of the materials array, same size as the volume.
#if NUM_MATERIALS > 0
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 0);
#endif
#if NUM_MATERIALS > 1
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 1);
#endif
#if NUM_MATERIALS > 2
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 2);
#endif
#if NUM_MATERIALS > 3
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 3);
#endif
#if NUM_MATERIALS > 4
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 4);
#endif
#if NUM_MATERIALS > 5
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 5);
#endif
#if NUM_MATERIALS > 6
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 6);
#endif
#if NUM_MATERIALS > 7
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 7);
#endif
#if NUM_MATERIALS > 8
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 8);
#endif
#if NUM_MATERIALS > 9
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 9);
#endif
#if NUM_MATERIALS > 10
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 10);
#endif
#if NUM_MATERIALS > 11
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 11);
#endif
#if NUM_MATERIALS > 12
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 12);
#endif
#if NUM_MATERIALS > 13
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 13);
#endif

#undef CURR_VOL_ID
#endif


/*** Handle eight volumes ***/
#if NUM_VOLUMES > 7
#define CURR_VOL_ID 7
// the CT volume
texture<float, 3, cudaReadModeElementType> VOLUME(CURR_VOL_ID);

// channel of the materials array, same size as the volume.
#if NUM_MATERIALS > 0
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 0);
#endif
#if NUM_MATERIALS > 1
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 1);
#endif
#if NUM_MATERIALS > 2
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 2);
#endif
#if NUM_MATERIALS > 3
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 3);
#endif
#if NUM_MATERIALS > 4
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 4);
#endif
#if NUM_MATERIALS > 5
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 5);
#endif
#if NUM_MATERIALS > 6
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 6);
#endif
#if NUM_MATERIALS > 7
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 7);
#endif
#if NUM_MATERIALS > 8
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 8);
#endif
#if NUM_MATERIALS > 9
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 9);
#endif
#if NUM_MATERIALS > 10
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 10);
#endif
#if NUM_MATERIALS > 11
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 11);
#endif
#if NUM_MATERIALS > 12
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 12);
#endif
#if NUM_MATERIALS > 13
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 13);
#endif

#undef CURR_VOL_ID
#endif


/*** Handle nine volumes ***/
#if NUM_VOLUMES > 8
#define CURR_VOL_ID 8
// the CT volume
texture<float, 3, cudaReadModeElementType> VOLUME(CURR_VOL_ID);

// channel of the materials array, same size as the volume.
#if NUM_MATERIALS > 0
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 0);
#endif
#if NUM_MATERIALS > 1
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 1);
#endif
#if NUM_MATERIALS > 2
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 2);
#endif
#if NUM_MATERIALS > 3
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 3);
#endif
#if NUM_MATERIALS > 4
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 4);
#endif
#if NUM_MATERIALS > 5
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 5);
#endif
#if NUM_MATERIALS > 6
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 6);
#endif
#if NUM_MATERIALS > 7
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 7);
#endif
#if NUM_MATERIALS > 8
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 8);
#endif
#if NUM_MATERIALS > 9
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 9);
#endif
#if NUM_MATERIALS > 10
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 10);
#endif
#if NUM_MATERIALS > 11
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 11);
#endif
#if NUM_MATERIALS > 12
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 12);
#endif
#if NUM_MATERIALS > 13
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 13);
#endif

#undef CURR_VOL_ID
#endif


/*** Handle ten volumes ***/
#if NUM_VOLUMES > 9
#define CURR_VOL_ID 9
// the CT volume
texture<float, 3, cudaReadModeElementType> VOLUME(CURR_VOL_ID);

// channel of the materials array, same size as the volume.
#if NUM_MATERIALS > 0
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 0);
#endif
#if NUM_MATERIALS > 1
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 1);
#endif
#if NUM_MATERIALS > 2
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 2);
#endif
#if NUM_MATERIALS > 3
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 3);
#endif
#if NUM_MATERIALS > 4
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 4);
#endif
#if NUM_MATERIALS > 5
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 5);
#endif
#if NUM_MATERIALS > 6
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 6);
#endif
#if NUM_MATERIALS > 7
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 7);
#endif
#if NUM_MATERIALS > 8
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 8);
#endif
#if NUM_MATERIALS > 9
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 9);
#endif
#if NUM_MATERIALS > 10
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 10);
#endif
#if NUM_MATERIALS > 11
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 11);
#endif
#if NUM_MATERIALS > 12
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 12);
#endif
#if NUM_MATERIALS > 13
texture<float, 3, cudaReadModeElementType> SEG(CURR_VOL_ID, 13);
#endif

#undef CURR_VOL_ID
#endif
