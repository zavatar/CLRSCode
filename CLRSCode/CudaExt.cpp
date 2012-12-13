//////////////////////////////////////////////////////////////////////////
//
//	CudaExt.cpp
//	
//	CUDA Extension.
//
//	Written by Meng Zhu ,Zhejiang University (zhumeng1989@gmail.com)
//	Creation date: Oct.30 2012, 18:36
//
//	All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////

#include "CudaExt.h"

#include <Windows.h>
#include <stdio.h>

namespace cuda {

void SafeCall(cudaError err) {
	if (cudaSuccess != err) {
		printf("%s(%i): Runtime API error %d: %s.\n", __FILE__, __LINE__,
			(int)err, cudaGetErrorString(err));
		exit(-1);
	}
} 

void PrintProp(cudaDeviceProp prop, int dev) {
	printf( " --- General Information for device %d ---\n", dev );
	printf( "Name: %s\n", prop.name );
	printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
	printf("  Clock rate: %d\n", prop.clockRate );
	printf("  Memory Clock Rate (KHz): %d\n",
		prop.memoryClockRate);
	printf("  Memory Bus Width (bits): %d\n",
		prop.memoryBusWidth);
	printf("  Peak Memory Bandwidth (GB/s): %f\n",
		2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
	printf( "Device copy overlap: " );
	if (prop.deviceOverlap)
		printf( "Enabled\n" );
	else
		printf( "Disabled\n" );
	printf( "Kernel execition timeout : " );
	if (prop.kernelExecTimeoutEnabled)
		printf( "Enabled\n" );
	else
		printf( "Disabled\n" );
	printf( "\n --- Memory Information for device %d ---\n", dev );
	printf( "Total global mem: 0x%08x\n", prop.totalGlobalMem );
	printf( "Total constant Mem: 0x%08x\n", prop.totalConstMem );
	printf( "Max mem pitch: 0x%08x\n", prop.memPitch );
	printf( "Texture Alignment: 0x%08x\n", prop.textureAlignment );
	printf( "\n --- MP Information for device %d ---\n", dev );
	printf( "Multiprocessor count: %d\n",
		prop.multiProcessorCount );
	printf( "Shared mem per mp: 0x%08x\n", prop.sharedMemPerBlock );
	printf( "Registers per mp: %d\n", prop.regsPerBlock );
	printf( "Threads in warp: %d\n", prop.warpSize );
	printf( "Max threads per block: %d\n",
		prop.maxThreadsPerBlock );
	printf( "Max thread dimensions: (%d, %d, %d)\n",
		prop.maxThreadsDim[0], prop.maxThreadsDim[1],
		prop.maxThreadsDim[2] );
	printf( "Max grid dimensions: (%d, %d, %d)\n",
		prop.maxGridSize[0], prop.maxGridSize[1],
		prop.maxGridSize[2] );
	printf( "\n" );
}

void InitDev()
{
	int dev(-1);
	cudaDeviceProp devProp;
	memset(&devProp, 0, sizeof(cudaDeviceProp));
	devProp.major = 1; devProp.minor = 3;
	SafeCall(cudaChooseDevice(&dev, &devProp));
	SafeCall(cudaGetDeviceProperties(&devProp, dev));
	SafeCall(cudaSetDevice(dev));
	PrintProp(devProp, dev);
}

Timer::Timer()
{
	SafeCall(cudaEventCreate(&_start));
	SafeCall(cudaEventCreate(&_stop));
}

void Timer::run()
{
	SafeCall(cudaEventRecord(_start, 0));
}

void Timer::stop()
{
	SafeCall(cudaEventRecord(_stop, 0));
}

float Timer::print()
{
	SafeCall(cudaEventSynchronize(_stop));
	float elapsedTime;
	SafeCall(cudaEventElapsedTime(&elapsedTime, _start, _stop));
	printf("GPU Kernel Time: %.3f ms\n", elapsedTime);
	return elapsedTime;
}

Timer::~Timer()
{
	SafeCall(cudaEventDestroy(_start));
	SafeCall(cudaEventDestroy(_stop));
}

}