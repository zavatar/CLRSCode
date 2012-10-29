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

#include <device_launch_parameters.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include <iostream>
#include <ctime>
#include <vector>
#include <algorithm>
#include <windows.h>

void ELAPSEDTIME(LARGE_INTEGER t1, LARGE_INTEGER t2);

namespace clrs {


void Thrust_sort_Benchmark(int LENGTH)
{
	LARGE_INTEGER t1, t2;
	std::cout<<"\nThrust_sort: \n";

	thrust::host_vector<float> h_vec(LENGTH);
	std::generate(h_vec.begin(), h_vec.end(), rand);

	QueryPerformanceCounter(&t1);

	// transfer data to the device
	thrust::device_vector<float> d_vec = h_vec;

	// sort data on the device (846M keys per second on GeForce GTX 480)
	thrust::sort(d_vec.begin(), d_vec.end());

	// transfer data back to host
	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

	QueryPerformanceCounter(&t2); ELAPSEDTIME(t1, t2);

	//ASSERTSORTED<ISPRINT>(&vA[0], LENGTH);
}

}