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
    output[(n)] += (multiplier) * tex3D(volume, px, py, pz) * round(cubicTex3D(seg(n), px, py, pz));\
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
        float *intensity, // flat array, with shape (out_height, out_width).
        float *photon_prob, // flat array, with shape (out_height, out_width).
        int n_bins, // the number of spectral bins
        float *energies, // 1-D array -- size is the n_bins
        float *pdf, // 1-D array -- probability density function over the energies
        float *absorb_coef_table, // flat [n_bins x NUM_MATERIALS] table that represents
                        // the precomputed get_absorbtion_coef values.
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

        // material projection-output channels
        float output[NUM_MATERIALS];

        // initialize the projection-output to 0.
        for (int m = 0; m < NUM_MATERIALS; m++) {
            output[m] = 0;
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

            // Perform the interpolation. This involves the variables: output, idx, px, py, pz, and volume. 
            // It is done for each segmentation.
            INTERPOLATE(boundary_factor);
        }

        // Scaling by step;
        output[0] *= step;

        // Last segment of the line
        if (output[0] > 0.0f) {
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

        // normalize output value to world coordinate system units
        for (int m = 0; m < NUM_MATERIALS; m++) {
            output[m] *= sqrt((rx * gVoxelElementSizeX)*(rx * gVoxelElementSizeX) + (ry * gVoxelElementSizeY)*(ry * gVoxelElementSizeY) + (rz * gVoxelElementSizeZ)*(rz * gVoxelElementSizeZ));
            
            // convert to centimeters
            output[m] /= 10;
        }

        /* Up to this point, we have accomplished the original projectKernel functionality.
         * The next steps to do are combining the forward_projections dictionary-ization and 
         * the mass_attenuation computation.
         * 
         * output[m] contains, for material 'm', the length (in centimeters) of the ray's path that passes 
         * through material 'm', multiplied by the density of the material (in g / cm^3).  Accordingly, the
         * units of output[m] are (g / cm^2).
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
         *      output[m] is the product of [linear distance of the ray through material 'm'] and 
         * [density of the material].  Accordingly, output[m] has units of [g / cm^2].
         *
         * The mass attenuation code uses the Beer-Lambert law:
         *
         *      I = I_{0} exp[-(\mu / \rho) * \rho * d]
         *
         * where I_{0} is the initial intensity, (\mu / \rho) is the mass attenuation coefficient, 
         * \rho is the density, and d is the length of the ray passing through the material.  Note 
         * that the product (\rho * d), also known as the 'area density' is the quantity output[m].
         *      Because we are attenuating multiple materials, the exponent that we use for the 
         * Beer-Lambert law is the sum of the (\mu_{mat} / \rho_{mat}) * (\rho_{mat} * d_{mat}) for
         * each material 'mat'.
         *
         *      The above explains the calculation up to and including 
         *              'intensity_tmp = expf(intensity_tmp)',
         * but does not yet explain the remaining calculation.  The remaining calculation serves to 
         * approximate the workings of a pixel in the dectector:
         *      
         *      pixelReading = \sum_{E} attenuatedBeamStrength[E] * E * p(E)
         *
         * where attenuatedBeamStrength follows the Beer-Lambert law as above, E is the energies of
         * the spectrum, and p(E) is the PDF of the spectrum.
         */
        for (int bin = 0; bin < n_bins; bin++) {
            float energy = energies[bin];
            float p = pdf[bin];

            float intensity_tmp = 0.0f; // lifting the call to calculate_attenuation_gpu(...) up a level
            for (int m = 0; m < NUM_MATERIALS; m++) {
                float absorb_coef = absorb_coef_table[bin * NUM_MATERIALS + m];
                intensity_tmp += output[m] * -1 * absorb_coef;
            }
            intensity_tmp = expf(intensity_tmp) * energy * p;
            // done with the "lifted" call to calculate_attenuation_gpu(...)

            intensity[img_dx] += intensity_tmp;
            photon_prob[img_dx] += intensity_tmp * (1.0 / energy);
        }

        return;
    }
}
    
