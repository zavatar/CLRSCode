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

#include "Benchmarks.h"

#include "CudaExt.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

namespace mythrust{

template <typename T>
void sortGPUtimer(T *l, T *r)
{
	cuda::Timer timer;
 	timer.run();

	sort(l, r);

	timer.stop();
 	timer.print();
}

template <typename T>
void sort(T *l, T *r)
{
	// transfer data to the device
	thrust::device_vector<T> d_vec(l, r);

	// sort data on the device (846M keys per second on GeForce GTX 480)
	thrust::sort(d_vec.begin(), d_vec.end());

	// transfer data back to host
	thrust::copy(d_vec.begin(), d_vec.end(), l);
}

}

void cudaMain()
{
	typedef bm::Type Type;

	bm::BenchMarks<Type, bm::LENGTH, bm::ISPRINT> benchmarks;

	benchmarks.sort(mythrust::sortGPUtimer<Type>, "\nThrust_sort: \n");

}
