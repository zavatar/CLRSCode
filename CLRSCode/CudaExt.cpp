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

Device::Device( int major, int minor )
{
	memset(&prop, 0, sizeof(cudaDeviceProp));
	initDevice(major, minor);
}

void Device::initDevice( int major, int minor )
{
	prop.major = major; prop.minor = minor;
	SafeCall(cudaChooseDevice(&dev, &prop));
	if (prop.major != major || prop.minor != minor) {
		printf("Choose Device error\n");
		exit(-1);
	}
	SafeCall(cudaGetDeviceProperties(&prop, dev));
	SafeCall(cudaSetDevice(dev));
	// Establish the context to avoid slow first call.
	// http://stackoverflow.com/questions/10415204/how-to-create-a-cuda-context
	cudaFree(0);
}

void Device::PrintProp()
{
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

void CheckDims( int bn, int tn )
{
	int dev;
	cudaDeviceProp prop;
	SafeCall(cudaGetDevice(&dev));
	SafeCall(cudaGetDeviceProperties(&prop, dev));
	if (bn > prop.maxGridSize[0] || tn > prop.maxThreadsPerBlock) {
		printf("Dimension error\n");
		exit(-1);
	}
}

}