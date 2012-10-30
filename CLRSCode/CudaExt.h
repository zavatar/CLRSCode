//////////////////////////////////////////////////////////////////////////
//	
//	CudaExt.h
//	
//	CUDA Extension.
//	
//	Written by Meng Zhu ,Zhejiang University (zhumeng1989@gmail.com)
//	Creation date: Apr.7 2012, 15:56
//  Updated: Oct.30 2012, 13:31
//
//	All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////

#ifndef CudaExt_H
#define CudaExt_H

#include <cuda_runtime.h>

namespace cuda {

void SafeCall(cudaError);

void PrintProp(cudaDeviceProp, int);

void InitDev();

class Timer {

public:
	Timer();

	void run();

	void stop();

	void print();

	~Timer();

private:

	cudaEvent_t _start, _stop;

};

} // namespace cuda

#endif