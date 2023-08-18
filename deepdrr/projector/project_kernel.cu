#include <cubicTex3D.cu>
#include <math_constants.h>
#include <stdio.h>

#ifndef AIR_DENSITY
#define AIR_DENSITY 0.1129
#endif

extern "C" {
__device__ static void calculate_solid_angle(const float *world_from_index, // (3, 3) array giving the world_from_index ray
                                                                      // transform for the camera
                                             float * __restrict__ solid_angle, // flat array, with shape (out_height, out_width).
                                             const int udx, // index into image width
                                             const int vdx, // index into image height
                                             const int img_dx // index into solid_angle
) {
    /**
     * SOLID ANGLE CALCULATION
     *
     * Let the pixel's four corners be c0, c1, c2, c3.  Split the pixel into two
     * right triangles.  These triangles each form a tetrahedron with the X-ray
     * source S.  We can then use a solid-angle-of-tetrahedron formula.
     *
     * From Wikipedia:
     *      Let OABC be the vertices of a tetrahedron with an origin at O
     * subtended by the triangular face ABC where \vec{a}, \vec{b}, \vec{c} are
     * the vectors \vec{SA}, \vec{SB}, \vec{SC} respectively.  Then,
     *
     * tan(\Omega / 2) = NUMERATOR / DENOMINATOR, with
     *
     * NUMERATOR = \vec{a} \cdot (\vec{b} \times \vec{c})
     * DENOMINATOR = abc + (\vec{a} \cdot \vec{b}) c + (\vec{a} \cdot \vec{c}) b
     * +
     * (\vec{b} \cdot \vec{c}) a
     *
     * where a,b,c are the magnitudes of their respective vectors.
     *
     * There are two potential pitfalls with the above formula.
     * 1. The NUMERATOR (a scalar triple product) can be negative if \vec{a},
     * \vec{b}, \vec{c} have the wrong winding.  Since no other portion of the
     * formula depends on the winding, computing the absolute value of the
     * scalar triple product is sufficient.
     * 2. If the NUMERATOR is positive but the DENOMINATOR is negative, the
     * formula returns a negative value that must be increased by \pi.
     */

    /*
     * PIXEL DIAGRAM
     *
     * corner0 __ corner1
     *        |__|
     * corner3    corner2
     */
    float cx[4]; // source-to-corner vector x-values in world space
    float cy[4]; // source-to-corner vector y-values in world space
    float cz[4]; // source-to-corner vector z-values in world space
    float cmag[4]; // magnitude of source-to-corner vector

    float cu_offset[4] = {0.f, 1.f, 1.f, 0.f};
    float cv_offset[4] = {0.f, 0.f, 1.f, 1.f};
    for (int c = 0; c < 4; c++) {
        float cu = udx + cu_offset[c];
        float cv = vdx + cv_offset[c];

        cx[c] = cu * world_from_index[0] + cv * world_from_index[1] + world_from_index[2];
        cy[c] = cu * world_from_index[3] + cv * world_from_index[4] + world_from_index[5];
        cz[c] = cu * world_from_index[6] + cv * world_from_index[7] + world_from_index[8];

        cmag[c] = sqrtf((cx[c] * cx[c]) + (cy[c] * cy[c]) + (cz[c] * cz[c]));
    }

    /*
     * The cross- and dot-products needed for the [c0, c1, c2] triangle are:
     *
     * - absolute value of triple product of c0,c1,c2 = c1 \cdot (c0 \times c2)
     *      Since the magnitude of the triple product is invariant under
     * reorderings of the three vectors, we choose to cross-product c0,c2 so we
     * can reuse that result
     * - dot product of c0, c1
     * - dot product of c0, c2
     * - dot product of c1, c2
     *
     * The products needed for the [c0, c2, c3] triangle are:
     *
     * - absolute value of triple product of c0,c2,c3 = c3 \cdot (c0 \times c2)
     *      Since the magnitude of the triple product is invariant under
     * reorderings of the three vectors, we choose to cross-product c0,c2 so we
     * can reuse that result
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

    float denom_012 =
        (cmag[0] * cmag[1] * cmag[2]) + (c0_dot_c1 * cmag[2]) + (c0_dot_c2 * cmag[1]) + (c1_dot_c2 * cmag[0]);
    float denom_023 =
        (cmag[0] * cmag[2] * cmag[3]) + (c0_dot_c2 * cmag[3]) + (c0_dot_c3 * cmag[2]) + (c2_dot_c3 * cmag[0]);

    float solid_angle_012 = 2.f * atan2(numer_012, denom_012);
    if (solid_angle_012 < 0.0f) {
        solid_angle_012 += CUDART_PI_F;
    }
    float solid_angle_023 = 2.f * atan2(numer_023, denom_023);
    if (solid_angle_023 < 0.0f) {
        solid_angle_023 += CUDART_PI_F;
    }

    solid_angle[img_dx] = solid_angle_012 + solid_angle_023;
}

__global__ void
projectKernel(const cudaTextureObject_t * __restrict__ volume_texs, // array of volume textures
              const cudaTextureObject_t * __restrict__ seg_texs, // array of segmentation textures
              const int out_width, // width of the output image
              const int out_height, // height of the output image
              const float step, // step size (TODO: in world)
              const int * __restrict__ priority, // volumes with smaller priority-ID have higher priority
                             // when determining which volume we are in
              const float * __restrict__ gVolumeEdgeMinPointX, // These give a bounding box in world-space around each volume.
              const float * __restrict__ gVolumeEdgeMinPointY, // These give a bounding box in world-space around each volume.
              const float * __restrict__ gVolumeEdgeMinPointZ, // These give a bounding box in world-space around each volume.
              const float * __restrict__ gVolumeEdgeMaxPointX, // These give a bounding box in world-space around each volume.
              const float * __restrict__ gVolumeEdgeMaxPointY, // These give a bounding box in world-space around each volume.
              const float * __restrict__ gVolumeEdgeMaxPointZ, // These give a bounding box in world-space around each volume.
              const float * __restrict__ gVoxelElementSizeX, // one value for each of the NUM_VOLUMES volumes
              const float * __restrict__ gVoxelElementSizeY, // one value for each of the NUM_VOLUMES volumes
              const float * __restrict__ gVoxelElementSizeZ, // one value for each of the NUM_VOLUMES volumes
              const float * __restrict__ sx_ijk, // x-coordinate of source point in IJK space for each volume (NUM_VOLUMES,)
              const float * __restrict__ sy_ijk, // y-coordinate of source point in IJK space for each volume (NUM_VOLUMES,)
              const float * __restrict__ sz_ijk, // z-coordinate of source point in IJK space for each
                             // volume (NUM_VOLUMES,) (passed in to avoid re-computing
                             // on every thread)
              const float max_ray_length, // max distance a ray can travel
              const float * __restrict__ world_from_index, // (3, 3) array giving the world_from_index ray transform for the camera
              const float * __restrict__ ijk_from_world, // (NUM_VOLUMES, 3, 4) transform giving the transform
                                     // from world to IJK coordinates for each volume.
              const int n_bins, // the number of spectral bins
              const float * __restrict__ energies, // 1-D array -- size is the n_bins. Units: [keV]
              const float * __restrict__ pdf, // 1-D array -- probability density function over the energies
              const float * __restrict__ absorb_coef_table, // flat [n_bins x NUM_MATERIALS] table that
                                        // represents the precomputed
                                        // get_absorption_coef values. index into the
                                        // table as: table[bin * NUM_MATERIALS + mat]
              float * __restrict__ intensity, // flat array, with shape (out_height, out_width).
              float * __restrict__ photon_prob, // flat array, with shape (out_height, out_width).
              float * __restrict__ solid_angle, // flat array, with shape (out_height, out_width). Could be NULL pointer
              const float * __restrict__ mesh_hit_alphas, // mesh hit distances for subtracting
              const int8_t * __restrict__ mesh_hit_facing, // mesh hit facing direction for subtracting
              const float * __restrict__ additive_densities, // additive densities
              const int * __restrict__ mesh_unique_materials, // unique materials for additive mesh
              const int mesh_unique_material_count, // number of unique materials for additive mesh
            //   const int max_mesh_depth, // maximum number of mesh hits per pixel
              const int offsetW, 
              const int offsetH) {
    // The output image has the following coordinate system, with cell-centered
    // sampling. y is along the fast axis (columns), x along the slow (rows).
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
    // int debug = (udx == 243) && (vdx == 149); // 4x4 binning

    // if (udx == 40) printf("udx: %d, vdx: %d\n", udx, vdx);

    // if the current point is outside the output image, no computation needed
    if (udx >= out_width || vdx >= out_height)
        return;

    // flat index to pixel in *intensity and *photon_prob
    // int img_dx = vdx * out_width + udx;
    int img_dx = (udx * out_height) + vdx;

    // initialize intensity and photon_prob to 0
    intensity[img_dx] = 0;
    photon_prob[img_dx] = 0;

    if (NULL != solid_angle) {
        calculate_solid_angle(world_from_index, solid_angle, udx, vdx, img_dx);
    }

    // cell-centered sampling point corresponding to pixel index, in
    // index-space.
    float u = (float)udx + 0.5;
    float v = (float)vdx + 0.5;

    // Vector in world-space along ray from source-point to pixel at [u,v] on
    // the detector plane.
    float rx = u * world_from_index[0] + v * world_from_index[1] + world_from_index[2];
    float ry = u * world_from_index[3] + v * world_from_index[4] + world_from_index[5];
    float rz = u * world_from_index[6] + v * world_from_index[7] + world_from_index[8];

    /* make the ray a unit vector */
    float ray_length = sqrtf(rx * rx + ry * ry + rz * rz);
    float inv_ray_norm = 1.0f / ray_length;
    rx *= inv_ray_norm;
    ry *= inv_ray_norm;
    rz *= inv_ray_norm;

    // calculate projections
    // Part 1: compute alpha value at entry and exit point of all volumes on
    // either side of the ray, in world-space. minAlpha: the distance from
    // source point to all-volumes entry point of the ray, in world-space.
    // maxAlpha: the distance from source point to all-volumes exit point of the
    // ray.
    float minAlpha = INFINITY; // the furthest along the ray we want to consider
                               // is the start point.
    float maxAlpha = 0; // closest point to consider is at the detector
    float minAlpha_vol[NUM_VOLUMES],
        maxAlpha_vol[NUM_VOLUMES]; // same, but just for each volume.
    float alpha0, alpha1, reci;
    int do_trace[NUM_VOLUMES]; // for each volume, whether or not to perform the
                               // ray-tracing
    int do_return = 1;

    // Get the ray direction in the IJK space for each volume.
    float rx_ijk[NUM_VOLUMES];
    float ry_ijk[NUM_VOLUMES];
    float rz_ijk[NUM_VOLUMES];
    int offs = 12; // TODO: fix bad style
    for (int i = 0; i < NUM_VOLUMES; i++) {
        // Homogeneous transform of a vector.
        rx_ijk[i] = ijk_from_world[offs * i + 0] * rx + ijk_from_world[offs * i + 1] * ry +
                    ijk_from_world[offs * i + 2] * rz + ijk_from_world[offs * i + 3] * 0;
        ry_ijk[i] = ijk_from_world[offs * i + 4] * rx + ijk_from_world[offs * i + 5] * ry +
                    ijk_from_world[offs * i + 6] * rz + ijk_from_world[offs * i + 7] * 0;
        rz_ijk[i] = ijk_from_world[offs * i + 8] * rx + ijk_from_world[offs * i + 9] * ry +
                    ijk_from_world[offs * i + 10] * rz + ijk_from_world[offs * i + 11] * 0;

        // Get the number of times the ijk ray can fit between the source and
        // the entry/exit points of this volume in *this* IJK space.
        do_trace[i] = 1;
        minAlpha_vol[i] = 0;
        maxAlpha_vol[i] = max_ray_length > 0 ? max_ray_length : INFINITY;
        if (0.0f != rx_ijk[i]) {
            reci = 1.0f / rx_ijk[i];
            alpha0 = (gVolumeEdgeMinPointX[i] - sx_ijk[i]) * reci;
            alpha1 = (gVolumeEdgeMaxPointX[i] - sx_ijk[i]) * reci;
            minAlpha_vol[i] = fmax(minAlpha_vol[i], fmin(alpha0, alpha1));
            maxAlpha_vol[i] = fmin(maxAlpha_vol[i], fmax(alpha0, alpha1));
        } else if (gVolumeEdgeMinPointX[i] > sx_ijk[i] || sx_ijk[i] > gVolumeEdgeMaxPointX[i]) {
            do_trace[i] = 0;
            continue;
        }
        if (0.0f != ry_ijk[i]) {
            reci = 1.0f / ry_ijk[i];
            alpha0 = (gVolumeEdgeMinPointY[i] - sy_ijk[i]) * reci;
            alpha1 = (gVolumeEdgeMaxPointY[i] - sy_ijk[i]) * reci;
            minAlpha_vol[i] = fmax(minAlpha_vol[i], fmin(alpha0, alpha1));
            maxAlpha_vol[i] = fmin(maxAlpha_vol[i], fmax(alpha0, alpha1));
        } else if (gVolumeEdgeMinPointY[i] > sy_ijk[i] || sy_ijk[i] > gVolumeEdgeMaxPointY[i]) {
            do_trace[i] = 0;
            continue;
        }
        if (0.0f != rz_ijk[i]) {
            reci = 1.0f / rz_ijk[i];
            alpha0 = (gVolumeEdgeMinPointZ[i] - sz_ijk[i]) * reci;
            alpha1 = (gVolumeEdgeMaxPointZ[i] - sz_ijk[i]) * reci;
            minAlpha_vol[i] = fmax(minAlpha_vol[i], fmin(alpha0, alpha1));
            maxAlpha_vol[i] = fmin(maxAlpha_vol[i], fmax(alpha0, alpha1));
        } else if (gVolumeEdgeMinPointZ[i] > sz_ijk[i] || sz_ijk[i] > gVolumeEdgeMaxPointZ[i]) {
            do_trace[i] = 0;
            continue;
        }
        do_return = 0;

        // Now, this is valid, since "how many times the ray can fit in the
        // distance" is equivalent to the distance in world space, since [rx,
        // ry, rz] is a unit vector.
        minAlpha = fmin(minAlpha, minAlpha_vol[i]);
        maxAlpha = fmax(maxAlpha, maxAlpha_vol[i]);
    }

    // Means none of the volumes have do_trace = 1.
    if (do_return) {
        return;
    }

    // printf("global min, max alphas: %f, %f\n", minAlpha, maxAlpha);

    // Part 2: Cast ray if it intersects any of the volumes
    int num_steps = ceil((maxAlpha - minAlpha) / step);
    // if (debug) printf("num_steps: %d\n", num_steps);

    // initialize the projection-output to 0.
    float area_density[NUM_MATERIALS];
    for (int m = 0; m < NUM_MATERIALS; m++) {
        area_density[m] = 0.0f;
    }

    float px[NUM_VOLUMES]; // voxel-space point
    float py[NUM_VOLUMES];
    float pz[NUM_VOLUMES];
    float alpha = minAlpha; // distance along the world space ray (alpha =
                            // minAlpha[i] + step * t)
    int curr_priority; // the priority at the location
    int n_vols_at_curr_priority; // how many volumes to consider at the location
    float seg_at_alpha[NUM_VOLUMES][NUM_MATERIALS];
    // if (debug) printf("start trace\n");

// #if MESH_ADDITIVE_AND_SUBTRACTIVE_ENABLED > 0
    int mesh_hit_depth = 0;
    int mesh_hit_index = 0;
    // int hit_arr_index = 0;

// #endif

    // Attenuate up to minAlpha, assuming it is filled with air.
    if (ATTENUATE_OUTSIDE_VOLUME) {
        area_density[AIR_INDEX] += (minAlpha / step) * AIR_DENSITY;
    }

    int asdf = (vdx * out_width + udx) * MAX_MESH_DEPTH;

    int facing_local[MAX_MESH_DEPTH]; // faster
    float alpha_local[MAX_MESH_DEPTH];

    for (int i = 0; i < MAX_MESH_DEPTH; i++) {
        facing_local[i] = mesh_hit_facing[asdf + i];
        alpha_local[i] = mesh_hit_alphas[asdf + i];
    }

    int priority_local[NUM_VOLUMES]; // faster maybe?

    for (int i = 0; i < NUM_VOLUMES; i++) {
        priority_local[i] = priority[i];
    }

    float sx_ijk_local[NUM_VOLUMES];
    float sy_ijk_local[NUM_VOLUMES];
    float sz_ijk_local[NUM_VOLUMES];

    for (int i = 0; i < NUM_VOLUMES; i++) {
        sx_ijk_local[i] = sx_ijk[i];
        sy_ijk_local[i] = sy_ijk[i];
        sz_ijk_local[i] = sz_ijk[i];
    }

    // trace (if doing the last segment separately, need to use num_steps - 1
    for (int t = 0; t < num_steps; t++) {
        for (int vol_id = 0; vol_id < NUM_VOLUMES; vol_id++) {
            px[vol_id] = sx_ijk_local[vol_id] + alpha * rx_ijk[vol_id] - 0.5f;
            py[vol_id] = sy_ijk_local[vol_id] + alpha * ry_ijk[vol_id] - 0.5f;
            pz[vol_id] = sz_ijk_local[vol_id] + alpha * rz_ijk[vol_id] - 0.5f;

            for (int mat_id = 0; mat_id < NUM_MATERIALS; mat_id++) {
                // TODO (liam): discuss: why use fancy cubicTex3D and then round it?
                seg_at_alpha[vol_id][mat_id] = tex3D<float>(seg_texs[vol_id * NUM_MATERIALS + mat_id], px[vol_id], py[vol_id], pz[vol_id]);
                // seg_at_alpha[vol_id][mat_id] = roundf(cubicTex3D<float>(seg_texs[vol_id * NUM_MATERIALS + mat_id], px[vol_id], py[vol_id], pz[vol_id]));
            }
        }

        curr_priority = NUM_VOLUMES;
        n_vols_at_curr_priority = 0;
        for (int i = 0; i < NUM_VOLUMES; i++) {
            if (0 == do_trace[i]) {
                continue;
            }

            // If alpha is outside the volume bounds, then we can skip this
            // volume.
            if ((alpha < minAlpha_vol[i]) || (alpha > maxAlpha_vol[i])) {
                continue;
            }

            // If all segmentation values are 0, then we can skip this volume.
            float any_seg = 0.0f;
            for (int m = 0; m < NUM_MATERIALS; m++) {
                any_seg += seg_at_alpha[i][m];
                if (any_seg > 0.0f) {
                    break;
                }
            }
            if (0.0f == any_seg) {
                continue;
            }

            // Calculate highest priority (lowest priority num) at this
            // location.
            if (priority_local[i] < curr_priority) {
                curr_priority = priority_local[i];
                n_vols_at_curr_priority = 1;
            } else if (priority_local[i] == curr_priority) {
                n_vols_at_curr_priority += 1;
            }
        }

        // bool mesh_hit_this_step = false;
        
#if MESH_ADDITIVE_AND_SUBTRACTIVE_ENABLED > 0
        while (true) {
        // for (int i = 0; i < 2; i++) {
            if ((mesh_hit_index < MAX_MESH_DEPTH && facing_local[mesh_hit_index] != 0 && alpha_local[mesh_hit_index] < alpha)){
                mesh_hit_depth += facing_local[mesh_hit_index];
                mesh_hit_index += 1;
            } else {
                break;
            }
        }

        // if (mesh_hit_depth) {
        //     mesh_hit_this_step = true; // TODO mesh priorities?
        // }
#endif

        // if (debug) printf("  got priority at alpha, num vols\n"); // This is
        // the one that seems to take a half a second.
        if (!mesh_hit_depth) {
            if (0 == n_vols_at_curr_priority) {
                // Outside the bounds of all volumes to trace. Use the default
                // AIR_DENSITY.
                if (ATTENUATE_OUTSIDE_VOLUME) {
                    area_density[AIR_INDEX] += AIR_DENSITY;
                }
            } else {
                // If multiple volumes at the same priority, use the average
                float weight = 1.0f / ((float)n_vols_at_curr_priority);

                // For the entry boundary, multiply by 0.5. That is, for the
                // initial interpolated value, only a half step-size is
                // considered in the computation. For the second-to-last
                // interpolation point, also multiply by 0.5, since there will
                // be a final step at the globalMaxAlpha boundary.
                weight *= (0 == t || num_steps - 1 == t) ? 0.5f : 1.0f;

                // Loop through volumes and add to the area_density.
                for (int vol_id = 0; vol_id < NUM_VOLUMES; vol_id++) {
                    if (do_trace[vol_id] && (priority_local[vol_id] == curr_priority)) {
                        float vol_density = tex3D<float>(volume_texs[vol_id], px[vol_id], py[vol_id], pz[vol_id]);
                        for (int mat_id = 0; mat_id < NUM_MATERIALS; mat_id++) {
                            area_density[mat_id] +=
                                (weight)*vol_density *
                                seg_at_alpha[vol_id][mat_id];
                        }
                    }
                }
            }
        }
        alpha += step;
    }

    // Attenuate from the end of the volume to the detector.
    if (ATTENUATE_OUTSIDE_VOLUME) {
        area_density[AIR_INDEX] += (ray_length - maxAlpha) / step * AIR_DENSITY;
    }

    // if (debug) printf("finished trace, num_steps: %d\n", num_steps);

    // Scaling by step
    for (int m = 0; m < NUM_MATERIALS; m++) {
        area_density[m] *= step;
    }

#if MESH_ADDITIVE_ENABLED > 0
    for (int i = 0; i < mesh_unique_material_count; i++) {
        int add_dens_idx = i * (out_height * out_width * 2) + (vdx * out_width + udx) * 2;
        // If there is a matching number of front and back hits, add the density
        if (fabs(additive_densities[add_dens_idx + 1]) < 0.00001) {
            area_density[mesh_unique_materials[i]] += additive_densities[add_dens_idx];
        }
    }
#endif

    // Convert to centimeters
    for (int m = 0; m < NUM_MATERIALS; m++) {
        area_density[m] /= 10.0f;
    }

    /* Up to this point, we have accomplished the original projectKernel
     * functionality. The next steps to do are combining the forward_projections
     * dictionary-ization and the mass_attenuation computation
     */

    // forward_projections dictionary-ization is implicit.

    // MASS ATTENUATION COMPUTATION

    /**
     * EXPLANATION OF THE PHYSICS/MATHEMATICS
     *
     *      The mass attenuation coefficient (found in absorb_coef_table) is:
     * \mu / \rho, where \mu is the linear attenuation coefficient, and \rho is
     * the mass density.  \mu has units of inverse length, and \rho has units of
     * mass/volume, so the mass attenuation coefficient has units of [cm^2 / g]
     *      area_density[m] is the product of [linear distance of the ray
     * through material 'm'] and [density of the material].  Accordingly,
     * area_density[m] has units of [g / cm^2].
     *
     * The mass attenuation code uses the Beer-Lambert law:
     *
     *      I = I_{0} exp[-(\mu / \rho) * \rho * d]
     *
     * where I_{0} is the initial intensity, (\mu / \rho) is the mass
     * attenuation coefficient, \rho is the density, and d is the length of the
     * ray passing through the material.  Note that the product (\rho * d), also
     * known as the 'area density' is the quantity area_density[m]. Because we
     * are attenuating multiple materials, the exponent that we use for the
     * Beer-Lambert law is the sum of the (\mu_{mat} / \rho_{mat}) * (\rho_{mat}
     * * d_{mat}) for each material 'mat'.
     *
     *      The above explains the calculation up to and including
     *              '____ = expf(-1 * beer_lambert_exp)',
     * but does not yet explain the remaining calculation.  The remaining
     * calculation serves to approximate the workings of a pixel in the
     * dectector:
     *
     *      pixelReading = \sum_{E} attenuatedBeamStrength[E] * E * p(E)
     *
     * where attenuatedBeamStrength follows the Beer-Lambert law as above, E is
     * the energies of the spectrum, and p(E) is the PDF of the spectrum. Note
     * also that the Beer-Lambert law deals with the quantity 'intensity', which
     * is related to the power transmitted through [unit area perpendicular to
     * the direction of travel]. Since the intensities mentioned in the
     * Beer-Lambert law are proportional to 1/[unit area], we can replace the
     * "intensity" calcuation with simply the energies involved.  Later
     * conversion to other physical quanities can be done outside of the kernel.
     */
    // if (debug)  printf("attenuation\n");

    for (int bin = 0; bin < n_bins; bin++) { // 151
        float beer_lambert_exp = 0.0f;
        for (int m = 0; m < NUM_MATERIALS; m++) {
            beer_lambert_exp += area_density[m] * absorb_coef_table[bin * NUM_MATERIALS + m];
        }
        float photon_prob_tmp = expf(-1.f * beer_lambert_exp) * pdf[bin]; // dimensionless value

        photon_prob[img_dx] += photon_prob_tmp;
        intensity[img_dx] += energies[bin] * photon_prob_tmp; // units: [keV] per unit photon to hit the pixel
    }

    // if (debug) printf("done with kernel thread\n");
    return;
}
}
