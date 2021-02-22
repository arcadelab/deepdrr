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

/* Compton data constant */
#define MAX_NSHELLS 30

extern "C" {
    typedef struct plane_surface {
        // plane vector (nx, ny, nz, d), where \vec{n} is the normal vector and d is the distance to the origin
        float nx, ny, nz, d;
        // 'surface origin': a point on the plane that is used as the reference point for the plane's basis vectors 
        float ori_x, ori_y, ori_z;
        // the two basis vectors
        float b1_x, b1_y, b1_z;
        float b2_x, b2_y, b2_z;
        // the bounds for the basis vector multipliers to stay within the surface's region on the plane
        float bound1_lo, bound1_hi;
        float bound2_lo, bound2_hi;
        // can we assume that the basis vectors orthogonal?
        int orthogonal;
    } plane_surface_t;

    typedef struct rng_seed {
        int x, y;
    } rng_seed_t;

    typedef struct rita {
        int n_gridpts;
        double x[n_gridpts];
        double y[n_gridpts];
        double a[n_gridpts];
        double b[n_gridpts];
    } rita_t;

    typedef struct compton_data {
        int nshells;
        float f[MAX_NSHELLS]; // number of electrons in each shell
        float ui[MAX_NSHELLS]; // ionization energy for each shell, in [eV]
        float jmc[MAX_NSHELLS]; // (J_{i,0} m_{e} c) for each shell i. Dimensionless.
    } compton_data_t;

    __global__ void initialization_stage(
        int detector_width, // size of detector in pixels 
        int detector_height,
        char *nominal_segmentation, // [0..2]-labeled segmentation obtained by thresholding: [-infty, -500, 300, infty]
        float sx, // x-coordinate of source in IJK
        float sy,
        float sz,
        float *rt_kinv, // (3, 3) array giving the image-to-world-ray transform.
        int n_bins, // the number of spectral bins
        float *spectrum_energies, // 1-D array -- size is the n_bins
        float *spectrum_cdf, // 1-D array -- cumulative density function over the energies
        int photon_count, // number of photons to simulate (emit from source)
        float E_abs, // the energy level below which photons are assumed to be absorbed
        float *deposited_energy // the output.  Size is [detector_width]x[detector_height]
    ) {
        // TODO: further develop the arguments
        return;
    }

    __device__ void move_photon_to_volume(
        float *pos_x, // position of the photon.  Serves as both input and ouput
        float *pos_y,
        float *pos_z,
        float dx, // direction of photon travel
        float dy,
        float dz,
        int *hits_volume, // Boolean output.  Does the photon actually hit the volume?
        float gVolumeEdgeMinPointX, // bounds of the volume
        float gVolumeEdgeMinPointY,
        float gVolumeEdgeMinPointZ,
        float gVolumeEdgeMaxPointX,
        float gVolumeEdgeMaxPointY,
        float gVolumeEdgeMaxPointZ,
        float gVoxelElementSizeX, // voxel size
        float gVoxelElementSizeY,
        float gVoxelElementSizeZ
    ) {
        // TODO: implement
        return;
    }

    __device__ void get_scattered_dir(
        float *dx, // both input and output
        float *dy,
        float *dz,
        double cos_theta,
        double phi
    ) {
        // TODO: implement
        return;
    }

    __device__ void sample_initial_dir(
        float *dx,
        float *dy,
        float *dz
    ) {
        // TODO: implement
        return;
    }

    __device__ float sample_initial_energy(
        const int n_bins,
        const float *spectrum_energies,
        const float *spectrum_cdf
    ) {
        // TODO: implement -- binary search?
        return 0.0f
    }

    __device__ double sample_rita(const rita_t *sampler) {
        // TODO: implement
        return 0.0;
    }

    __device__ double sample_Rayleigh(
        float energy,
        const rita_t *ff_sampler
    ) {
        double cos_theta = 0.0f;
        // TODO: implement
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