/*
 * Based on Sisniega et al. (2015), "High-fidelity artifact correction for cone-beam CT imaging of the brain"
 */

 #include "scatter.h"

extern "C" {
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
        float gVoxelElementSizeX, // voxel size in IJK
        float gVoxelElementSizeY,
        float gVoxelElementSizeZ,
        float *index_from_ijk, // (2, 3) array giving the IJK-homogeneous-coord.s-to-pixel-coord.s transformation
        mat_mfp_data_t *mfp_data_arr,
        wc_mfp_data_t *woodcock_mfp,
        compton_data_t *compton_arr,
        rita_t *rita_arr,
        plane_surface_t *detector_plane,
        int n_bins, // the number of spectral bins
        float *spectrum_energies, // 1-D array -- size is the n_bins. Units: [keV]
        float *spectrum_cdf, // 1-D array -- cumulative density function over the energies
        float E_abs, // the energy level below which photons are assumed to be absorbed [keV]
        int seed_input,
        float *deposited_energy, // the output.  Size is [detector_width]x[detector_height]
        int *scattered_hits, // number of scattered photons that hit the detector
        int *unscattered_hits // number of unscattered photons that hit the detector
    );

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
    );

    __device__ int get_voxel_1D(
        float3_t *pos,
        float3_t *gVolumeEdgeMinPoint,
        float3_t *gVolumeEdgeMaxPoint,
        float3_t *gVoxelElementSize,
        int3_t *volume_shape
    );

    __device__ void get_scattered_dir(
        float3_t *dir, // direction: both input and output
        double cos_theta, // polar scattering angle
        double phi // azimuthal scattering angle
    );

    __device__ void move_photon_to_volume(
        float3_t *pos, // position of the photon.  Serves as both input and ouput
        float3_t *dir, // input: direction of photon travel
        int *hits_volume, // Boolean output.  Does the photon actually hit the volume?
        float3_t *gVolumeEdgeMinPoint, // IJK coordinate of minimum bounds of volume
        float3_t *gVolumeEdgeMaxPoint, // IJK coordinate of maximum bounds of volume
    );

    __device__ void sample_initial_dir(
        float3_t *dir, // output: the sampled direction
        rng_seed_t *seed
    );

    __device__ float sample_initial_energy(
        const int n_bins,
        const float *spectrum_energies,
        const float *spectrum_cdf,
        rng_seed_t *seed
    );

    __device__ double sample_rita(
        const rita_t *sampler,
        rng_seed_t *seed
    );

    __device__ float psurface_check_ray_intersection(
        float3_t *pos, // input: current position of the photon
        float3_t *dir, // input: direction of photon travel
        const plane_surface_t *psur
    );

    __device__ void get_mat_mfp_data(
        mat_mfp_data_t *data,
        float nrg, // energy of the photon
        float *ra, // output: MFP for Rayleigh scatter. Units: [mm]
        float *co, // output: MFP for Compton scatter. Units: [mm]
        float *tot // output: MFP (total). Units: [mm]
    );

    __device__ void get_wc_mfp_data(
        wc_mfp_data_t *data,
        float nrg, // energy of the photon [eV]
        float *mfp // output: Woodcock MFP. Units: [mm]
    );

    __device__ double sample_Rayleigh(
        float energy,
        const rita_t *ff_sampler,
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