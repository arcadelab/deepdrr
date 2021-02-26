#include <stdio.h>
#include <cubicTex3D.cu>

#ifndef NUM_MATERIALS
#define NUM_MATERIALS 14
#endif

#define _seg(n) seg_##n
#define seg(n) _seg(n)

// channel of the materials array, same size as the volume.
#if NUM_MATERIALS > 0
texture<float, 3, cudaReadModeElementType> seg(0);
#endif
#if NUM_MATERIALS > 1
texture<float, 3, cudaReadModeElementType> seg(1);
#endif
#if NUM_MATERIALS > 2
texture<float, 3, cudaReadModeElementType> seg(2);
#endif
#if NUM_MATERIALS > 3
texture<float, 3, cudaReadModeElementType> seg(3);
#endif
#if NUM_MATERIALS > 4
texture<float, 3, cudaReadModeElementType> seg(4);
#endif
#if NUM_MATERIALS > 5
texture<float, 3, cudaReadModeElementType> seg(5);
#endif
#if NUM_MATERIALS > 6
texture<float, 3, cudaReadModeElementType> seg(6);
#endif
#if NUM_MATERIALS > 7
texture<float, 3, cudaReadModeElementType> seg(7);
#endif
#if NUM_MATERIALS > 8
texture<float, 3, cudaReadModeElementType> seg(8);
#endif
#if NUM_MATERIALS > 9
texture<float, 3, cudaReadModeElementType> seg(9);
#endif
#if NUM_MATERIALS > 10
texture<float, 3, cudaReadModeElementType> seg(10);
#endif
#if NUM_MATERIALS > 11
texture<float, 3, cudaReadModeElementType> seg(11);
#endif
#if NUM_MATERIALS > 12
texture<float, 3, cudaReadModeElementType> seg(12);
#endif
#if NUM_MATERIALS > 13
texture<float, 3, cudaReadModeElementType> seg(13);
#endif

#define UPDATE(multiplier, n) do {\
    area_density[(n)] += (multiplier) * tex3D(volume, px, py, pz) * round(cubicTex3D(seg(n), px, py, pz));\
} while (0)

#if NUM_MATERIALS == 1
#define INTERPOLATE(multiplier) do {\
    UPDATE(multiplier, 0);\
} while (0)
#elif NUM_MATERIALS == 2
#define INTERPOLATE(multiplier) do {\
    UPDATE(multiplier, 0);\
    UPDATE(multiplier, 1);\
} while (0)
#elif NUM_MATERIALS == 3
#define INTERPOLATE(multiplier) do {\
    UPDATE(multiplier, 0);\
    UPDATE(multiplier, 1);\
    UPDATE(multiplier, 2);\
} while (0)
#elif NUM_MATERIALS == 4
#define INTERPOLATE(multiplier) do {\
    UPDATE(multiplier, 0);\
    UPDATE(multiplier, 1);\
    UPDATE(multiplier, 2);\
    UPDATE(multiplier, 3);\
} while (0)
#elif NUM_MATERIALS == 5
#define INTERPOLATE(multiplier) do {\
    UPDATE(multiplier, 0);\
    UPDATE(multiplier, 1);\
    UPDATE(multiplier, 2);\
    UPDATE(multiplier, 3);\
    UPDATE(multiplier, 4);\
} while (0)
#elif NUM_MATERIALS == 6
#define INTERPOLATE(multiplier) do {\
    UPDATE(multiplier, 0);\
    UPDATE(multiplier, 1);\
    UPDATE(multiplier, 2);\
    UPDATE(multiplier, 4);\
    UPDATE(multiplier, 5);\
} while (0)
#elif NUM_MATERIALS == 7
#define INTERPOLATE(multiplier) do {\
    UPDATE(multiplier, 0);\
    UPDATE(multiplier, 1);\
    UPDATE(multiplier, 2);\
    UPDATE(multiplier, 4);\
    UPDATE(multiplier, 5);\
    UPDATE(multiplier, 6);\
} while (0)
#elif NUM_MATERIALS == 8
#define INTERPOLATE(multiplier) do {\
    UPDATE(multiplier, 0);\
    UPDATE(multiplier, 1);\
    UPDATE(multiplier, 2);\
    UPDATE(multiplier, 4);\
    UPDATE(multiplier, 5);\
    UPDATE(multiplier, 6);\
    UPDATE(multiplier, 7);\
} while (0)
#elif NUM_MATERIALS == 9
#define INTERPOLATE(multiplier) do {\
    UPDATE(multiplier, 0);\
    UPDATE(multiplier, 1);\
    UPDATE(multiplier, 2);\
    UPDATE(multiplier, 4);\
    UPDATE(multiplier, 5);\
    UPDATE(multiplier, 6);\
    UPDATE(multiplier, 7);\
    UPDATE(multiplier, 8);\
} while (0)
#elif NUM_MATERIALS == 10
#define INTERPOLATE(multiplier) do {\
    UPDATE(multiplier, 0);\
    UPDATE(multiplier, 1);\
    UPDATE(multiplier, 2);\
    UPDATE(multiplier, 4);\
    UPDATE(multiplier, 5);\
    UPDATE(multiplier, 6);\
    UPDATE(multiplier, 7);\
    UPDATE(multiplier, 8);\
    UPDATE(multiplier, 9);\
} while (0)
#elif NUM_MATERIALS == 11
#define INTERPOLATE(multiplier) do {\
    UPDATE(multiplier, 0);\
    UPDATE(multiplier, 1);\
    UPDATE(multiplier, 2);\
    UPDATE(multiplier, 4);\
    UPDATE(multiplier, 5);\
    UPDATE(multiplier, 6);\
    UPDATE(multiplier, 7);\
    UPDATE(multiplier, 8);\
    UPDATE(multiplier, 9);\
    UPDATE(multiplierl, 10);\
} while (0)
#elif NUM_MATERIALS == 12
#define INTERPOLATE(multiplier) do {\
    UPDATE(multiplier, 0);\
    UPDATE(multiplier, 1);\
    UPDATE(multiplier, 2);\
    UPDATE(multiplier, 4);\
    UPDATE(multiplier, 5);\
    UPDATE(multiplier, 6);\
    UPDATE(multiplier, 7);\
    UPDATE(multiplier, 8);\
    UPDATE(multiplier, 9);\
    UPDATE(multiplier, 10);\
    UPDATE(multiplier, 11);\
} while (0)
#elif NUM_MATERIALS == 13
#define INTERPOLATE(multiplier) do {\
    UPDATE(multiplier, 0);\
    UPDATE(multiplier, 1);\
    UPDATE(multiplier, 2);\
    UPDATE(multiplier, 4);\
    UPDATE(multiplier, 5);\
    UPDATE(multiplier, 6);\
    UPDATE(multiplier, 7);\
    UPDATE(multiplier, 8);\
    UPDATE(multiplier, 9);\
    UPDATE(multiplier, 10);\
    UPDATE(multiplier, 11);\
    UPDATE(multiplier, 12);\
} while (0)
#elif NUM_MATERIALS == 14
#define INTERPOLATE(multiplier) do {\
    UPDATE(multiplier, 0);\
    UPDATE(multiplier, 1);\
    UPDATE(multiplier, 2);\
    UPDATE(multiplier, 4);\
    UPDATE(multiplier, 5);\
    UPDATE(multiplier, 6);\
    UPDATE(multiplier, 7);\
    UPDATE(multiplier, 8);\
    UPDATE(multiplier, 9);\
    UPDATE(multiplier, 10);\
    UPDATE(multiplier, 11);\
    UPDATE(multiplier, 12);\
    UPDATE(multiplier, 13);\
} while (0)
#else
#define INTERPOLATE(multiplier) do {\
    fprintf(stderr, "NUM_MATERIALS not in [1, 14]");\
} while (0)
#endif

// the CT volume (used to be tex_density)
texture<float, 3, cudaReadModeElementType> volume;

#define PI_FLOAT  3.14159265358979323846f

extern "C" {
    __global__  void projectKernel(
        int out_width, // width of the output image
        int out_height, // height of the output image
        float step,
        float gVolumeEdgeMinPointX,
        float gVolumeEdgeMinPointY,
        float gVolumeEdgeMinPointZ,
        float gVolumeEdgeMaxPointX,
        float gVolumeEdgeMaxPointY,
        float gVolumeEdgeMaxPointZ,
        float gVoxelElementSizeX,
        float gVoxelElementSizeY,
        float gVoxelElementSizeZ,
        float sx, // x-coordinate of source point for rays in world-space
        float sy,
        float sz,
        float *rt_kinv, // (3, 3) array giving the image-to-world-ray transform.
        int photon_count, // total number of photons emitted by X-ray source
        float *deposited_energy, // flat array, with shape (out_height, out_width).
        float *photon_prob, // flat array, with shape (out_height, out_width).
        int n_bins, // the number of spectral bins
        float *energies, // 1-D array -- size is the n_bins
        float *pdf, // 1-D array -- probability density function over the energies
        float *absorb_coef_table, // flat [n_bins x NUM_MATERIALS] table that represents
                        // the precomputed get_absorption_coef values.
                        // index into the table as: table[bin * NUM_MATERIALS + mat]
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
        float rx = u * rt_kinv[0] + v * rt_kinv[1] + rt_kinv[2];
        float ry = u * rt_kinv[3] + v * rt_kinv[4] + rt_kinv[5];
        float rz = u * rt_kinv[6] + v * rt_kinv[7] + rt_kinv[8];

        // make the ray a unit vector
        float normFactor = 1.0f / (sqrt((rx * rx) + (ry * ry) + (rz * rz)));
        rx *= normFactor;
        ry *= normFactor;
        rz *= normFactor;

        // calculate projections
        // Part 1: compute alpha value at entry and exit point of the volume on either side of the ray.
        // minAlpha: the distance from source point to volume entry point of the ray.
        // maxAlpha: the distance from source point to volume exit point of the ray.
        float minAlpha, maxAlpha;
        minAlpha = 0;
        maxAlpha = INFINITY;

        if (0.0f != rx)
        {
            float reci = 1.0f / rx;
            float alpha0 = (gVolumeEdgeMinPointX - sx) * reci;
            float alpha1 = (gVolumeEdgeMaxPointX - sx) * reci;
            minAlpha = fmin(alpha0, alpha1);
            maxAlpha = fmax(alpha0, alpha1);
        }
        else if (gVolumeEdgeMinPointX > sx || sx > gVolumeEdgeMaxPointX)
        {
            return;
        }

        if (0.0f != ry)
        {
            float reci = 1.0f / ry;
            float alpha0 = (gVolumeEdgeMinPointY - sy) * reci;
            float alpha1 = (gVolumeEdgeMaxPointY - sy) * reci;
            minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
            maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
        }
        else if (gVolumeEdgeMinPointY > sy || sy > gVolumeEdgeMaxPointY)
        {
            return;
        }

        if (0.0f != rz)
        {
            float reci = 1.0f / rz;
            float alpha0 = (gVolumeEdgeMinPointZ - sz) * reci;
            float alpha1 = (gVolumeEdgeMaxPointZ - sz) * reci;
            minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
            maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
        }
        else if (gVolumeEdgeMinPointZ > sz || sz > gVolumeEdgeMaxPointZ)
        {
            return;
        }

        // we start not at the exact entry point 
        // => we can be sure to be inside the volume
        // (this is commented out intentionally, seemingly)
        //minAlpha += step * 0.5f;
        
        // Part 2: Cast ray if it intersects the volume

        // Trapezoidal rule (interpolating function = piecewise linear func)
        float px, py, pz; // voxel-space point
        int t; // number of steps along ray
        float alpha; // distance along ray (alpha = minAlpha + step * t)
        float boundary_factor; // factor to multiply at the boundary.

        // Output channels of the raycasting
        // Stores the product of [linear distance of the ray through material 'm'] and 
        // [density of the material]
        float area_density[NUM_MATERIALS];

        // initialize the raycasting output (the area density) to 0.
        for (int m = 0; m < NUM_MATERIALS; m++) {
            area_density[m] = 0;
        }

        // Sample the points along the ray at the entrance boundary of the volume and the mid segments.
        for (t = 0, alpha = minAlpha; alpha < maxAlpha; t++, alpha += step)
        {
            // Get the current sample point in the volume voxel-space.
            // In CUDA, voxel centeras are located at (xx.5, xx.5, xx.5), whereas SwVolume has voxel centers at integers.
            px = sx + alpha * rx + 0.5;
            py = sy + alpha * ry + 0.5;
            pz = sz + alpha * rz - gVolumeEdgeMinPointZ;

            /* For the entry boundary, multiply by 0.5 (this is the t == 0 check). That is, for the initial interpolated value, 
             * only a half step-size is considered in the computation.
             * For the second-to-last interpolation point, also multiply by 0.5, since there will be a final step at the maxAlpha boundary.
             */ 
            boundary_factor = (t == 0 || alpha + step >= maxAlpha) ? 0.5 : 1.0;

            // Perform the interpolation. This involves the variables: area_density, idx, px, py, pz, and volume. 
            // It is done for each segmentation.
            INTERPOLATE(boundary_factor);
        }

        // Scaling by step;
        for (int m = 0; m < NUM_MATERIALS; m++) {
            area_density[m] *= step;
        }

        // Last segment of the line
        if (area_density[0] > 0.0f) {
            alpha -= step;
            float lastStepsize = maxAlpha - alpha;

            // scaled last step interpolation (something weird?)
            INTERPOLATE(0.5 * lastStepsize);

            // The last segment of the line integral takes care of the varying length.
            px = sx + alpha * rx + 0.5;
            py = sy + alpha * ry + 0.5;
            pz = sz + alpha * rz - gVolumeEdgeMinPointZ;

            // interpolation
            INTERPOLATE(0.5 * lastStepsize);
        }

        // normalize area_density value to world coordinate system units
        for (int m = 0; m < NUM_MATERIALS; m++) {
            area_density[m] *= sqrt((rx * gVoxelElementSizeX)*(rx * gVoxelElementSizeX) + (ry * gVoxelElementSizeY)*(ry * gVoxelElementSizeY) + (rz * gVoxelElementSizeZ)*(rz * gVoxelElementSizeZ));
            
            // convert to centimeters, since the ray-cast length is in voxels, AKA millimeters
            area_density[m] /= 10;
        }

        /* Up to this point, we have accomplished the original projectKernel functionality.
         * The next steps to do are combining the forward_projections dictionary-ization and 
         * the mass_attenuation computation.
         * 
         * area_density[m] contains, for material 'm', the length (in centimeters) of the ray's path that passes 
         * through material 'm', multiplied by the density of the material (in g / cm^3).  Accordingly, the
         * units of area_density[m] are (g / cm^2).
         */

        // forward_projections dictionary-ization is implicit.

        // flat index to pixel in *deposited_energy and *photon_prob
        int img_dx = (udx * out_height) + vdx;

        // zero-out deposited_energy and photon_prob
        deposited_energy[img_dx] = 0;
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
         * true (physical) intensity, by dividing by the pixel area, can be done outside of the kernel.
         */
        for (int bin = 0; bin < n_bins; bin++) {
            float energy = energies[bin];
            float p = pdf[bin];

            float beer_lambert_exp = 0.0f;
            for (int m = 0; m < NUM_MATERIALS; m++) {
                beer_lambert_exp += area_density[m] * absorb_coef_table[bin * NUM_MATERIALS + m];
            }
            float photon_prob_tmp = expf(-1 * beer_lambert_exp) * p; // dimensionless value

            photon_prob[img_dx] += photon_prob_tmp;
            deposited_energy[img_dx] += energy * photon_prob_tmp; // units: [eV] per unit photon to hit the pixel
        }

        /**
         * At this point, deposited_energy contains the quantity:
         * [energy deposited on the pixel per unit photon that hits the pixel]
         * 
         * To get the true amount of energy deposited on the pixel, we need to scale up
         * by the factor:
         *      [number of photons that hit the detector] 
         *              = [photon_count] x [solid angle covered by the pixel]
         * where [photon_count] is the total number of photons emitted by the X-ray source
         * in all directions, and [solid angle covered by the pixel] can be calculated as 
         * described below.
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

        float solid_angle_012 = (numer_012 / denom_012);
        if (solid_angle_012 < 0.0f) {
            solid_angle_012 += PI_FLOAT;
        }
        float solid_angle_023 = (numer_023 / denom_023);
        if (solid_angle_023 < 0.0f) {
            solid_angle_023 += PI_FLOAT;
        }

        /*if ((0 == udx)) {
            printf("solid angle of pixel [%d, %d]: %1.10e\n", udx, vdx, (numer_012 / denom_012) + (numer_023 / denom_023));
        }*/

        // Scale up deposited_energy
        deposited_energy[img_dx] *= ((float)photon_count) * ((numer_012 / denom_012) + (numer_023 / denom_023));

        return;
    }
}
    
