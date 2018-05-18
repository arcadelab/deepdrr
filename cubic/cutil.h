#ifndef _CUTIL_MINI_H_
#define _CUTIL_MINI_H_

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#include <assert.h>
#include <vector>
#include <cuda_gl_interop.h>
#include <helper_timer.h>

static inline bool CUT_DEVICE_INIT(int argc, char** argv)
{
	cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
	{
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return false;
    }

	return true;
}

static inline void CUDA_SAFE_CALL(cudaError_t err)
{
	assert(err == cudaSuccess);
}

enum CUTBoolean 
{
    CUTFalse = 0,
    CUTTrue = 1
};

static inline void CUT_SAFE_CALL(CUTBoolean result)
{
	assert(result == CUTTrue);
}

static inline void CUT_CHECK_ERROR(const char* errorMessage)
{
	cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
	{
        fprintf(stderr, "Cuda error: %s: %s.\n", errorMessage, cudaGetErrorString(err) );
    }
}

static std::vector<StopWatchInterface*> __timers;
static CUTBoolean cutCreateTimer(unsigned int* name)
{
	StopWatchInterface* timer;
	sdkCreateTimer(&timer);
	__timers.push_back(timer);
	*name = __timers.size() - 1;
	return CUTTrue;
}

static CUTBoolean cutResetTimer(const unsigned int name)
{
	sdkResetTimer(&__timers[name]);
	return CUTTrue;
}

static CUTBoolean cutStartTimer(const unsigned int name)
{
	sdkStartTimer(&__timers[name]);
	return CUTTrue;
}

static CUTBoolean cutStopTimer(const unsigned int name)
{
	sdkStopTimer(&__timers[name]);
	return CUTTrue;
}

static float cutGetTimerValue(const unsigned int name)
{
	return sdkGetTimerValue(&__timers[name]);
}

#endif //_CUTIL_MINI_H_
