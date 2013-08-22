//////////////////////////////////////////////////////////////////////////
//
//	ThrustBenchmarks.cu
//
//	Written by Meng Zhu ,Zhejiang University (zhumeng1989@gmail.com)
//	Creation date: Oct.29 2012, 22:37
//
//	All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////

#include "CudaExt.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

namespace mythrust{

template <typename T>
void sort(T *l, T *r)
{
	// transfer data to the device
	thrust::device_vector<T> d_vec(l, r);

	cuda::Timer timer;
 	timer.run();
	// sort data on the device (846M keys per second on GeForce GTX 480)
	thrust::sort(d_vec.begin(), d_vec.end());
	timer.stop();
 	
	// transfer data back to host
	thrust::copy(d_vec.begin(), d_vec.end(), l);
	timer.print();
}

template
void sort<float>(float *l, float *r);

template
void sort<int>(int *l, int *r);

}
