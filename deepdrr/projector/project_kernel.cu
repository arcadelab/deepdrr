
#include <stdio.h>
#include <cubicTex3D.cu>

#ifndef NUM_MATERIALS
#define NUM_MATERIALS 3
#endif

// the CT volume (used to be tex_density)
texture<float, 3, cudaReadModeElementType> volume;

//  channel of the materials array, same size as the volume.
texture<float, 3, cudaReadModeElementType> materials_0;
texture<float, 3, cudaReadModeElementType> materials_1;
texture<float, 3, cudaReadModeElementType> materials_2;

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
        float* gInvARmatrix, // (3, 3) array giving the image-to-world-ray transform.
        float* output, // flat array, with shape (out_height, out_width, NUM_MATERIALS).
        int offsetW,
        int offsetH)
    {
        int widx = threadIdx.x + (blockIdx.x + offsetW) * blockDim.x; // index into output width
        int hidx = threadIdx.y + (blockIdx.y + offsetH) * blockDim.y; // index into output height

        // if the current point is outside the output image, no computation needed
        if (widx >= out_width || hidx >= out_height)
            return;

        // flat index to first material in output "channel". 
        // So (idx + m) gets you the pixel for material index m in [0, NUM_MATERIALS)
        int idx = widx * (out_height * NUM_MATERIALS) + hidx * NUM_MATERIALS; 

        // image-space point corresponding to pixel
        float u = (float) widx + 0.5;
        float v = (float) hidx + 0.5;

        // vector along world-space ray from source-point to pixel on the image plane
        float rx = gInvARmatrix[2] + v * gInvARmatrix[1] + u * gInvARmatrix[0];
        float ry = gInvARmatrix[5] + v * gInvARmatrix[4] + u * gInvARmatrix[3];
        float rz = gInvARmatrix[8] + v * gInvARmatrix[7] + u * gInvARmatrix[6];

        // make the ray a unit-vector
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
        float px, py, pz; // world-space point
        int t; // number of steps along ray
        int m; // materials index
        float alpha; // distance along ray (alpha = minAlpha + step * t)
        float boundary_factor; // factor to multiply at the boundary.

        // initialize the output to 0.
        for (m = 0; m < NUM_MATERIALS; m++) {
            output[idx + m] = 0;
        }

        // Sample the points along the ray at the entrance boundary of the volume and the mid segments.
        for (t = 0, alpha = minAlpha; alpha < maxAlpha; t++, alpha += step)
        {
            // Get the current sample point in the volume world-space.
            // In CUDA, voxel centeras are located at (xx.5, xx.5, xx.5), whereas SwVolume has voxel centers at integers.
            px = sx + alpha * rx + 0.5;
            py = sy + alpha * ry + 0.5;
            pz = sz + alpha * rz - gVolumeEdgeMinPointZ;

            /* For the entry boundary, multiply by 0.5 (this is the i == 0 check). That is, for the initial interpolated value, 
             * only a half step-size is considered in the computation.
             * For the second-to-last interpolation point, also multiply by 0.5, since there will be a final step at the maxAlpha boundary.
             */ 
            boundary_factor = (t == 0 || alpha + step >= maxAlpha) ? 0.5 : 1.0;

            // Perform the interpolation.
            output[idx + 0] += boundary_factor * tex3D(volume, px, py, pz) * round(cubicTex3D(materials_0, px, py, pz));
            output[idx + 1] += boundary_factor * tex3D(volume, px, py, pz) * round(cubicTex3D(materials_1, px, py, pz));
            output[idx + 2] += boundary_factor * tex3D(volume, px, py, pz) * round(cubicTex3D(materials_2, px, py, pz));
        }

        // Scaling by step;
        output[idx] *= step;

        // Last segment of the line
        if (output[idx] > 0.0f ) {
            alpha -= step;
            float lastStepsize = maxAlpha - alpha;

            // scaled last step interpolation (something weird?)
            output[idx + 0] += 0.5 * lastStepsize * tex3D(volume, px, py, pz) * round(cubicTex3D(materials_0, px, py, pz));
            output[idx + 1] += 0.5 * lastStepsize * tex3D(volume, px, py, pz) * round(cubicTex3D(materials_1, px, py, pz));
            output[idx + 2] += 0.5 * lastStepsize * tex3D(volume, px, py, pz) * round(cubicTex3D(materials_2, px, py, pz));

            // The last segment of the line integral takes care of the varying length.
            px = sx + alpha * rx + 0.5;
            py = sy + alpha * ry + 0.5;
            pz = sz + alpha * rz - gVolumeEdgeMinPointZ;

            // interpolation
            output[idx + 0] += 0.5 * lastStepsize * tex3D(volume, px, py, pz) * round(cubicTex3D(materials_0, px, py, pz));
            output[idx + 1] += 0.5 * lastStepsize * tex3D(volume, px, py, pz) * round(cubicTex3D(materials_1, px, py, pz));
            output[idx + 2] += 0.5 * lastStepsize * tex3D(volume, px, py, pz) * round(cubicTex3D(materials_2, px, py, pz));
        }

        // normalize output value to world coordinate system units
        output[idx] *= sqrt((rx * gVoxelElementSizeX)*(rx * gVoxelElementSizeX) + (ry * gVoxelElementSizeY)*(ry * gVoxelElementSizeY) + (rz * gVoxelElementSizeZ)*(rz * gVoxelElementSizeZ));
    
        return;
    }
}
    
