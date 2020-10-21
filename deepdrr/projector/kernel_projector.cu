
#include <stdio.h>
#include <cubicTex3D.cu>

texture<float, 3, cudaReadModeElementType> tex_density;
texture<float, 3, cudaReadModeElementType> tex_segmentation;
extern "C" {
    __global__  void projectKernel(
        int proj_width,
        int proj_height,
        float stepsize,
        float gVolumeEdgeMinPointX,
        float gVolumeEdgeMinPointY,
        float gVolumeEdgeMinPointZ,
        float gVolumeEdgeMaxPointX,
        float gVolumeEdgeMaxPointY,
        float gVolumeEdgeMaxPointZ,
        float gVoxelElementSizeX,
        float gVoxelElementSizeY,
        float gVoxelElementSizeZ,
        float sx,
        float sy,
        float sz,
        float* gInvARmatrix,
        float* output,
        int offsetW,
        int offsetH)
    {
        int udx = threadIdx.x + (blockIdx.x + offsetW) * blockDim.x;
        int vdx = threadIdx.y + (blockIdx.y + offsetH) * blockDim.y;
        int idx = udx*proj_height + vdx;

        if (udx >= proj_width || vdx >= proj_height) {
            return;}
        float u = (float) udx + 0.5;
        float v = (float) vdx + 0.5;

        // compute ray direction
        float rx = gInvARmatrix[2] + v * gInvARmatrix[1] + u * gInvARmatrix[0];
        float ry = gInvARmatrix[5] + v * gInvARmatrix[4] + u * gInvARmatrix[3];
        float rz = gInvARmatrix[8] + v * gInvARmatrix[7] + u * gInvARmatrix[6];

        // normalize ray direction float
        float normFactor = 1.0f / (sqrt((rx * rx) + (ry * ry) + (rz * rz)));
        rx *= normFactor;
        ry *= normFactor;
        rz *= normFactor;

        //calculate projections
        // Step 1: compute alpha value at entry and exit point of the volume
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
        //minAlpha += stepsize * 0.5f;
        
        // Step 2: Cast ray if it intersects the volume

        // Trapezoidal rule (interpolating function = piecewise linear func)
        float px, py, pz;

        // Entrance boundary
        // In CUDA, voxel centers are located at (xx.5, xx.5, xx.5),
        //  whereas, SwVolume has voxel centers at integers.
        // For the initial interpolated value, only a half stepsize is
        //  considered in the computation.
        if (minAlpha < maxAlpha) {
            px = sx + minAlpha * rx;
            py = sy + minAlpha * ry;
            pz = sz + minAlpha * rz;
            output[idx] += 0.5 * tex3D(tex_density, px + 0.5, py + 0.5, pz - gVolumeEdgeMinPointZ) * round(cubicTex3D(tex_segmentation, px + 0.5, py + 0.5, pz - gVolumeEdgeMinPointZ));
            minAlpha += stepsize;
        }

        // Mid segments
        while (minAlpha < maxAlpha)
        {
            px = sx + minAlpha * rx;
            py = sy + minAlpha * ry;
            pz = sz + minAlpha * rz;
            output[idx] += tex3D(tex_density, px + 0.5, py + 0.5, pz - gVolumeEdgeMinPointZ) * round(cubicTex3D(tex_segmentation, px + 0.5, py + 0.5, pz - gVolumeEdgeMinPointZ));
            minAlpha += stepsize;
        }

        // Scaling by stepsize;
        output[idx] *= stepsize;

        // Last segment of the line
        if (output[idx] > 0.0f ) {
            output[idx] -= 0.5 * stepsize * tex3D(tex_density, px + 0.5, py + 0.5, pz - gVolumeEdgeMinPointZ) * round(cubicTex3D(tex_segmentation, px + 0.5, py + 0.5, pz - gVolumeEdgeMinPointZ));
            minAlpha -= stepsize;
            float lastStepsize = maxAlpha - minAlpha;
            output[idx] += 0.5 * lastStepsize * tex3D(tex_density, px + 0.5, py + 0.5, pz - gVolumeEdgeMinPointZ) * round(cubicTex3D(tex_segmentation, px + 0.5, py + 0.5, pz - gVolumeEdgeMinPointZ));

            px = sx + maxAlpha * rx;
            py = sy + maxAlpha * ry;
            pz = sz + maxAlpha * rz;
            // The last segment of the line integral takes care of the
            // varying length.
            output[idx] += 0.5 * lastStepsize * tex3D(tex_density, px + 0.5, py + 0.5, pz - gVolumeEdgeMinPointZ) * round(cubicTex3D(tex_segmentation, px + 0.5, py + 0.5, pz - gVolumeEdgeMinPointZ));
        }

        // normalize output value to world coordinate system units
        output[idx] *= sqrt((rx * gVoxelElementSizeX)*(rx * gVoxelElementSizeX) + (ry * gVoxelElementSizeY)*(ry * gVoxelElementSizeY) + (rz * gVoxelElementSizeZ)*(rz * gVoxelElementSizeZ));
    
        return;
    }
}
    
