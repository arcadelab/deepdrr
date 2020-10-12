/*--------------------------------------------------------------------------*\
Copyright (c) 2008-2010, Danny Ruijters. All rights reserved.
http://www.dannyruijters.nl/cubicinterpolation/
This file is part of CUDA Cubic B-Spline Interpolation (CI).

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
*  Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
*  Neither the name of the copyright holders nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are
those of the authors and should not be interpreted as representing official
policies, either expressed or implied.

When using this code in a scientific project, please cite one or all of the
following papers:
*  Daniel Ruijters and Philippe Thévenaz,
   GPU Prefilter for Accurate Cubic B-Spline Interpolation, 
   The Computer Journal, vol. 55, no. 1, pp. 15-20, January 2012.
   http://dannyruijters.nl/docs/cudaPrefilter3.pdf
*  Daniel Ruijters, Bart M. ter Haar Romeny, and Paul Suetens,
   Efficient GPU-Based Texture Interpolation using Uniform B-Splines,
   Journal of Graphics Tools, vol. 13, no. 4, pp. 61-69, 2008.
\*--------------------------------------------------------------------------*/

#ifndef _MEMCPY_CUDA_H_
#define _MEMCPY_CUDA_H_

#include <stdio.h>
#include <cutil.h>
#include "internal/math_func.cu"

//--------------------------------------------------------------------------
// Declare the typecast CUDA kernels
//--------------------------------------------------------------------------
template<class T> __device__ float Multiplier()	{ return 1.0f; }
template<> __device__ float Multiplier<uchar>()	{ return 255.0f; }
template<> __device__ float Multiplier<schar>()	{ return 127.0f; }
template<> __device__ float Multiplier<ushort>(){ return 65535.0f; }
template<> __device__ float Multiplier<short>()	{ return 32767.0f; }

template<class T> __global__ void CopyCast(uchar* destination, const T* source, uint pitch, uint width)
{
	uint2 index = make_uint2(
		__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
		__umul24(blockIdx.y, blockDim.y) + threadIdx.y);

	float* dest = (float*)(destination + index.y * pitch) + index.x;
	*dest = (1.0f/Multiplier<T>()) * (float)(source[index.y * width + index.x]);
}

template<class T> __global__ void CopyCastBack(T* destination, const uchar* source, uint pitch, uint width)
{
	uint2 index = make_uint2(
		__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
		__umul24(blockIdx.y, blockDim.y) + threadIdx.y);

	float* src = (float*)(source + index.y * pitch) + index.x;
	destination[index.y * width + index.x] = (T)(Multiplier<T>() * *src);
}


//--------------------------------------------------------------------------
// Declare the typecast templated function
// This function can be called directly in C++ programs
//--------------------------------------------------------------------------

//! Allocate GPU memory and copy a voxel volume from CPU to GPU memory
//! and cast it to the normalized floating point format
//! @return the pointer to the GPU copy of the voxel volume
//! @param host  pointer to the voxel volume in CPU (host) memory
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
template<class T> extern cudaPitchedPtr CastVolumeHostToDevice(const T* host, uint width, uint height, uint depth)
{
	cudaPitchedPtr device = {0};
	const cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
	CUDA_SAFE_CALL(cudaMalloc3D(&device, extent));
	const size_t pitchedBytesPerSlice = device.pitch * device.ysize;
	
	T* temp = 0;
	const uint voxelsPerSlice = width * height;
	const size_t nrOfBytesTemp = voxelsPerSlice * sizeof(T);
	CUDA_SAFE_CALL(cudaMalloc((void**)&temp, nrOfBytesTemp));

	uint dimX = min(PowTwoDivider(width), 64);
	dim3 dimBlock(dimX, min(PowTwoDivider(height), 512 / dimX));
	dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);
	size_t offsetHost = 0;
	size_t offsetDevice = 0;
	
	for (uint slice = 0; slice < depth; slice++)
	{
		CUDA_SAFE_CALL(cudaMemcpy(temp, host + offsetHost, nrOfBytesTemp, cudaMemcpyHostToDevice));
		CopyCast<T><<<dimGrid, dimBlock>>>((uchar*)device.ptr + offsetDevice, temp, (uint)device.pitch, width);
		CUT_CHECK_ERROR("Cast kernel failed");
		offsetHost += voxelsPerSlice;
		offsetDevice += pitchedBytesPerSlice;
	}

	CUDA_SAFE_CALL(cudaFree(temp));  //free the temp GPU volume
	return device;
}

//! Copy a voxel volume from GPU to CPU memory
//! while casting it to the desired format
//! @param host  pointer to the voxel volume in CPU (host) memory
//! @param device  pitched pointer to the voxel volume in GPU (device) memory
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
//! @note The \host CPU memory should be pre-allocated
template<class T> extern void CastVolumeDeviceToHost(T* host, const cudaPitchedPtr device, uint width, uint height, uint depth)
{
	T* temp = 0;
	const uint voxelsPerSlice = width * height;
	const size_t nrOfBytesTemp = voxelsPerSlice * sizeof(T);
	CUDA_SAFE_CALL(cudaMalloc((void**)&temp, nrOfBytesTemp));

	uint dimX = min(PowTwoDivider(width), 64);
	dim3 dimBlock(dimX, min(PowTwoDivider(height), 512 / dimX));
	dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);
	const size_t pitchedBytesPerSlice = device.pitch * device.ysize;
	size_t offsetHost = 0;
	size_t offsetDevice = 0;
	
	for (uint slice = 0; slice < depth; slice++)
	{
		CopyCastBack<T><<<dimGrid, dimBlock>>>(temp, (const uchar*)device.ptr + offsetDevice, (uint)device.pitch, width);
		CUT_CHECK_ERROR("Cast kernel failed");
		CUDA_SAFE_CALL(cudaMemcpy(host + offsetHost, temp, nrOfBytesTemp, cudaMemcpyDeviceToHost));
		offsetHost += voxelsPerSlice;
		offsetDevice += pitchedBytesPerSlice;
	}

	CUDA_SAFE_CALL(cudaFree(temp));  //free the temp GPU volume
}

//--------------------------------------------------------------------------
// Copy floating point data from and to the GPU
//--------------------------------------------------------------------------

//! Allocate GPU memory and copy a voxel volume from CPU to GPU memory
//! @return the pitched pointer to the GPU copy of the voxel volume
//! @param host  pointer to the voxel volume in CPU (host) memory
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
extern "C"
cudaPitchedPtr CopyVolumeHostToDevice(const float* host, uint width, uint height, uint depth)
{
	cudaPitchedPtr device = {0};
	const cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
	CUDA_SAFE_CALL(cudaMalloc3D(&device, extent));
	cudaMemcpy3DParms p = {0};
	p.srcPtr = make_cudaPitchedPtr((void*)host, width * sizeof(float), width, height);
	p.dstPtr = device;
	p.extent = extent;
	p.kind = cudaMemcpyHostToDevice;
	CUDA_SAFE_CALL(cudaMemcpy3D(&p));
	return device;
}

//! Copy a voxel volume from GPU to CPU memory, and free the GPU memory
//! @param host  pointer to the voxel volume copy in CPU (host) memory
//! @param device  pitched pointer to the voxel volume in GPU (device) memory
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
//! @note The \host CPU memory should be pre-allocated
extern "C"
void CopyVolumeDeviceToHost(float* host, const cudaPitchedPtr device, uint width, uint height, uint depth)
{
	const cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
	cudaMemcpy3DParms p = {0};
	p.srcPtr = device;
	p.dstPtr = make_cudaPitchedPtr((void*)host, width * sizeof(float), width, height);
	p.extent = extent;
	p.kind = cudaMemcpyDeviceToHost;
	CUDA_SAFE_CALL(cudaMemcpy3D(&p));
	CUDA_SAFE_CALL(cudaFree(device.ptr));  //free the GPU volume
}

//! Copy a voxel volume from a pitched pointer to a texture
//! @param tex      [output]  pointer to the texture
//! @param texArray [output]  pointer to the texArray
//! @param volume   [input]   pointer to the the pitched voxel volume
//! @param extent   [input]   size (width, height, depth) of the voxel volume
//! @param onDevice [input]   boolean to indicate whether the voxel volume resides in GPU (true) or CPU (false) memory
//! @note When the texArray is not yet allocated, this function will allocate it
template<class T, enum cudaTextureReadMode mode> void CreateTextureFromVolume(
	texture<T, 3, mode>* tex, cudaArray** texArray,
	const cudaPitchedPtr volume, cudaExtent extent, bool onDevice)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
	if (*texArray == 0) CUDA_SAFE_CALL(cudaMalloc3DArray(texArray, &channelDesc, extent));
	// copy data to 3D array
	cudaMemcpy3DParms p = {0};
	p.extent   = extent;
	p.srcPtr   = volume;
	p.dstArray = *texArray;
	p.kind     = onDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
	CUDA_SAFE_CALL(cudaMemcpy3D(&p));
	// bind array to 3D texture
	CUDA_SAFE_CALL(cudaBindTextureToArray(*tex, *texArray, channelDesc));
	tex->normalized = false;  //access with absolute texture coordinates
	tex->filterMode = cudaFilterModeLinear;
}

//! Copy a voxel volume from continuous memory to a texture
//! @param tex      [output]  pointer to the texture
//! @param texArray [output]  pointer to the texArray
//! @param volume   [input]   pointer to the continuous memory with the voxel
//! @param extent   [input]   size (width, height, depth) of the voxel volume
//! @param onDevice [input]   boolean to indicate whether the voxel volume resides in GPU (true) or CPU (false) memory
//! @note When the texArray is not yet allocated, this function will allocate it
template<class T, enum cudaTextureReadMode mode> void CreateTextureFromVolume(
	texture<T, 3, mode>* tex, cudaArray** texArray,
	const T* volume, cudaExtent extent, bool onDevice)
{
	cudaPitchedPtr ptr = make_cudaPitchedPtr((void*)volume, extent.width*sizeof(T), extent.width, extent.height);
	CreateTextureFromVolume(tex, texArray, ptr, extent, onDevice);
}

#endif  //_MEMCPY_CUDA_H_
