/*
 * Based on Sisniega et al. (2015), "High-fidelity artifact correction for cone-beam CT imaging of the brain"
 */

#include <vector_types.h>

// Some typedefs for my personal preference
typedef int2 int2_t;
typedef int3 int3_t;
typedef float2 float2_t;
typedef float3 float3_t;

/*** DEFINES ***/
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

extern "C" {
    /*** STRUCT DEFINITIONS ***/
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

    typedef struct rayleigh_data {
        // RITA portion
        double x[MAX_RITA_N_PTS];
        double y[MAX_RITA_N_PTS];
        double a[MAX_RITA_N_PTS];
        double b[MAX_RITA_N_PTS];
        int n_gridpts;
        // Form factor data 
        float pmax[MAX_MFP_BINS];
    } rayleigh_data_t;
    
    typedef struct compton_data {
        int nshells;
        float f[MAX_NSHELLS]; // number of electrons in each shell
        float ui[MAX_NSHELLS]; // ionization energy for each shell, in [eV]
        float jmc[MAX_NSHELLS]; // (J_{i,0} m_{e} c) for each shell i. Dimensionless.
    } compton_data_t;
    
    // TODO: refactor the structs so that the energies don't have to be stored explicitly. xref MCGPU struct linear_interp
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

    /*** FUNCTION DECLARATIONS ***/
    __global__ void simulate_scatter(
        int detector_width, // size of detector in pixels 
        int detector_height,
        int histories_for_thread, // number of photons for -this- thread to track
        char *labeled_segmentation, // [0..NUM_MATERIALS-1]-labeled segmentation
        float sx, // coordinates of source in IJK
        float sy, // (not in a float3_t for ease of calling from Python wrapper)
        float sz,
        float sdd, // source-to-detector distance [mm]
        int volume_shape_x, // integer size of the volume to avoid 
        int volume_shape_y, // floating-point errors with the gVolumeEdge{Min,Max}Point
        int volume_shape_z, // and gVoxelElementSize math
        float gVolumeEdgeMinPointX, // bounds of the volume in IJK
        float gVolumeEdgeMinPointY,
        float gVolumeEdgeMinPointZ,
        float gVolumeEdgeMaxPointX,
        float gVolumeEdgeMaxPointY,
        float gVolumeEdgeMaxPointZ,
        float gVoxelElementSizeX, // voxel size in world
        float gVoxelElementSizeY,
        float gVoxelElementSizeZ,
        float *index_from_ijk, // (2, 3) array giving the IJK-homogeneous-coord.s-to-pixel-coord.s transformation
        mat_mfp_data_t *mfp_data_arr,
        wc_mfp_data_t *woodcock_mfp,
        compton_data_t *compton_arr,
        rayleigh_data_t *rayleigh_arr,
        plane_surface_t *detector_plane,
	float *world_from_ijk, // 3x4 transform
	float *ijk_from_world, // 3x4 transform
        int n_bins, // the number of spectral bins
        float *spectrum_energies, // 1-D array -- size is the n_bins. Units: [keV]
        float *spectrum_cdf, // 1-D array -- cumulative density function over the energies
        float E_abs, // the energy level below which photons are assumed to be absorbed [keV]
        int seed_input,
        float *deposited_energy, // the output.  Size is [detector_width]x[detector_height]
        int *num_scattered_hits, // number of scattered photons that hit the detector at each pixel. Same size as deposited_energy.
        int *num_unscattered_hits // number of unscattered photons that hit the detector at each pixel. Same size as deposited_energy.
    );

    __device__ void track_photon(
        float3_t *pos, // input: initial position in volume. output: end position of photon history
        float3_t *dir, // input: initial direction
        float *energy, // input: initial energy. output: energy at end of photon history. Units: [eV]
        int *hits_detector, // Boolean output.  Does the photon actually reach the detector plane?
        int *num_scatter_events, // should be passed a pointer to an int initialized to zero.  Returns the number of scatter events experienced by the photon
        float E_abs, // the energy level below which the photon is assumed to be absorbed. Units: [eV]
        char *labeled_segmentation, // [0..NUM_MATERIALS-1]-labeled segmentation
        mat_mfp_data_t *mfp_data_arr, // NUM_MATERIALS-element array of pointers to mat_mfp_data_t structs. Material associations based on labeled_segmentation
        wc_mfp_data_t *wc_data,
        compton_data_t *compton_arr, // NUM_MATERIALS-element array of pointers to compton_data_t.  Material associations as with mfp_data_arr
        rayleigh_data_t *rayleigh_arr, // NUM_MATERIALS-element array of pointers to rayleigh_t.  Material associations as with mfp_data_arr
        int3_t *volume_shape, // number of voxels in each direction IJK
        float3_t *gVolumeEdgeMinPoint, // IJK coordinate of minimum bounds of volume
        float3_t *gVolumeEdgeMaxPoint, // IJK coordinate of maximum bounds of volume
        plane_surface_t *detector_plane, 
	float *world_from_ijk, // 3x4 transform
	float *ijk_from_world, // 3x4 transform
        rng_seed_t *seed
    );

    __device__ int get_voxel_1D(
        float3_t *pos,
        float3_t *gVolumeEdgeMinPoint,
        float3_t *gVolumeEdgeMaxPoint,
        int3_t *volume_shape
    );

    __device__ void get_scattered_dir(
        float3_t *dir, // direction: both input and output
        double cos_theta, // polar scattering angle
        double phi, // azimuthal scattering angle
        float *world_from_ijk, // 3x4 transformation matrix
        float *ijk_from_world // 3x4 transformation matrix
    );

    __device__ void move_photon_to_volume(
        float3_t *pos, // position of the photon.  Serves as both input and ouput
        float3_t *dir, // input: direction of photon travel
        int *hits_volume, // Boolean output.  Does the photon actually hit the volume?
        float3_t *gVolumeEdgeMinPoint, // IJK coordinate of minimum bounds of volume
        float3_t *gVolumeEdgeMaxPoint  // IJK coordinate of maximum bounds of volume
    );

    __device__ void sample_initial_dir_world(
        float3_t *dir, // output: the sampled direction
        rng_seed_t *seed
    );

    __device__ void shift_point_frame_3x4_transform(
        float3_t *pt, // [in/out]: the point to be transformed
        float *transform
    );

    __device__ void shift_vector_frame_3x4_transform(
        float3_t *vec, // [in/out]: the vector to be transformed
        float *transform
    );

    __device__ float sample_initial_energy(
        const int n_bins,
        const float *spectrum_energies,
        const float *spectrum_cdf,
        rng_seed_t *seed
    );

    __device__ double sample_rita(
        const rayleigh_data_t *sampler,
        const double pmax_current,
        rng_seed_t *seed
    );

    __device__ float psurface_check_ray_intersection(
        float3_t *pos, // input: current position of the photon
        float3_t *dir, // input: direction of photon travel
        const plane_surface_t *psur
    );

    __device__ int find_energy_index(
        float nrg, 
        float *energy_arr, 
        int lo_idx,
        int hi_idx
    );

    __device__ void get_mat_mfp_data(
        mat_mfp_data_t *data,
        float nrg, // energy of the photon
        int e_index, // the index of the lower bound of the energy interval 
        float *ra, // output: MFP for Rayleigh scatter. Units: [mm]
        float *co, // output: MFP for Compton scatter. Units: [mm]
        float *tot // output: MFP (total). Units: [mm]
    );

    __device__ void get_wc_mfp_data(
        wc_mfp_data_t *data,
        float nrg, // energy of the photon [eV]
        int e_index, // the index of the lower bound of the energy interval 
        float *mfp // output: Woodcock MFP. Units: [mm]
    );

    __device__ double sample_Rayleigh(
        float energy,
        int e_index, // the index of the lower bound of the energy interval 
        const rayleigh_data_t *rayleigh_data,
        rng_seed_t *seed
    );

    __device__ double sample_Compton(
        float *energy, // serves as both input and output
        const compton_data_t *compton_data,
        rng_seed_t *seed
    );

    __device__ float ranecu(rng_seed_t *seed);
    __device__ double ranecu_double(rng_seed_t *seed);

    __device__ void initialize_seed(
        int thread_id, // each CUDA thread should have a unique ID given to it
        int histories_for_thread, 
        int seed_input,
        rng_seed_t *seed
    );

    __device__ int abMODm(
        int m,
        int a1, 
        int a2
    );
}
