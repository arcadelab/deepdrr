/*
 * Based on Sisniega et al. (2015), "High-fidelity artifact correction for cone-beam CT imaging of the brain"
 */

#define ELECTRON_REST_ENERGY 510998.918f // [eV]
#define INV_ELECTRON_REST_ENERGY 1.956951306108245e-6f // [eV]^{-1}

/* Material IDs for the nominal segmentation */
#define NOM_SEG_AIR_ID ((char) 0)
#define NOM_SEG_SOFT_ID ((char) 1)
#define NOM_SEG_BONE_ID ((char) 2)

/* Nominal density values, [g/cm^3] */
#define NOM_DENSITY_AIR 0.0f
#define NOM_DENSITY_SOFT 1.0f
#define NOM_DENSITY_BONE 1.92f

/* Data meta-constants */
#define MAX_NSHELLS 30
#define MAX_MFP_BINS 25005
#define MAX_RITA_N_PTS 128

/* Numerically necessary evils */
#define VOXEL_EPS      0.000015f // epsilon (small distance) that we use to ensure that 
#define NEG_VOXEL_EPS -0.000015f // the particle fully inside a voxel. Value from MC-GPU

/* Mathematical constants -- credit to Wolfram Alpha */
#define PI_FLOAT  3.14159265358979323846f
#define PI_DOUBLE 3.14159265358979323846
#define TWO_PI_FLOAT  6.28318530717958647693f
#define TWO_PI_DOUBLE 6.28318530717958647693

#define INFTY 500000.0f // inspired by MC-GPU :)
#define NEG_INFTY -500000.0f

/* Useful macros */
#define MAX_VAL(a, b) (((a) > (b)) ? (a) : (b))
#define MIN_VAL(a, b) (((a) < (b)) ? (a) : (b))

/*** STRUCT DEFINITIONS ***/
typedef struct int2 {
    int x, y;
} int2_t;

typedef struct int3 {
    int x, y, z;
} int3_t;

typedef struct float2 {
    float x, y;
} float2_t;

typedef struct float3 {
    float x, y, z;
} float3_t;

typedef struct plane_surface {
    // plane vector (nx, ny, nz, d), where \vec{n} is the normal vector and d is the distance to the origin
    float3_t n;
    float d;
    // 'surface origin': a point on the plane that is used as the reference point for the plane's basis vectors 
    float3_t ori;
    // the two basis vectors
    float3_t b1, b2;
    // the bounds for the basis vector multipliers to stay within the surface's region on the plane
    float2_t bound1, bound2; // .x is lower bound, .y is upper bound
    // can we assume that the basis vectors orthogonal?
    int orthogonal;
} plane_surface_t;

typedef struct rng_seed {
    int x, y;
} rng_seed_t;

typedef struct rita {
    int n_gridpts;
    double x[MAX_RITA_N_PTS];
    double y[MAX_RITA_N_PTS];
    double a[MAX_RITA_N_PTS];
    double b[MAX_RITA_N_PTS];
} rita_t;

typedef struct compton_data {
    int nshells;
    float f[MAX_NSHELLS]; // number of electrons in each shell
    float ui[MAX_NSHELLS]; // ionization energy for each shell, in [eV]
    float jmc[MAX_NSHELLS]; // (J_{i,0} m_{e} c) for each shell i. Dimensionless.
} compton_data_t;

typedef struct mat_mfp_data {
    int n_bins;
    float energy[MAX_MFP_BINS]; // Units: [eV]
    float mfp_Ra[MAX_MFP_BINS]; // Units: [mm]
    float mfp_Co[MAX_MFP_BINS]; // Units: [mm]
    float mfp_Tot[MAX_MFP_BINS]; // Units: [mm]
} mat_mfp_data_t;

typedef struct wc_mfp_data {
    int n_bins;
    float energy[MAX_MFP_BINS]; // Units: [eV]
    float mfp_wc[MAX_MFP_BINS]; // Units: [mm]
} wc_mfp_data_t;
