//////////////////////////////////////////////////////////////////////////
//
//	ParallelForall.h
//  https://developer.nvidia.com/content/easy-introduction-cuda-c-and-c
//
//  _K: kernel functon, _D: device function
//
//	Written by Meng Zhu ,Zhejiang University (zhumeng1989@gmail.com)
//	Creation date: Oct.29 2012, 20:03
//
//	All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////

#ifndef ParallelForall_H
#define ParallelForall_H

#include "CudaExt.h"

extern "C"
bool isPow2(unsigned int x)
{
	return ((x&(x-1))==0);
}

namespace pf {

void SAXPY_D(int nb, int nt, int n, float a, float *x, float *y);

void SAXPY()
{
	printf("\nSAXPY: \n");

	int N = 16*(1<<20);
	int nt = 512;
	int nb = (N + nt - 1)/nt;
	cuda::CheckDims(nb, nt);
	float *x, *y, *d_x, *d_y;
	x = (float*)malloc(N*sizeof(float));
	y = (float*)malloc(N*sizeof(float));

	cudaMalloc(&d_x, N*sizeof(float)); 
	cudaMalloc(&d_y, N*sizeof(float));

	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	cuda::Timer timer;

	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

	timer.run();
	// Perform SAXPY on 1M elements
	SAXPY_D(nb, nt, N, 2.f, d_x, d_y);
	timer.stop();

	cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

	float milliseconds=timer.print();

	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = std::max(maxError, abs(y[i]-4.0f));
	printf("Max error: %fn\n", maxError);

	printf("Effective Bandwidth (GB/s): %f\n", N*4*3/milliseconds/1e6);

	free(x); free(y);
	cudaFree(d_x); cudaFree(d_y);
}

template<class T>
T reduceCPU(T *data, int size)
{
	T sum = data[0];
	T c = (T)0.0;
	for (int i = 1; i < size; i++)
	{
		T y = data[i] - c;
		T t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}
	return sum;
}

unsigned int nextPow2(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

void gettnbn(int &bn, int &tn, int n)
{
	int dev;
	cudaDeviceProp prop;
	cuda::SafeCall(cudaGetDevice(&dev));
	cuda::SafeCall(cudaGetDeviceProperties(&prop, dev));

	int maxtn = 256, maxbn = 64;
	tn = (n < maxtn*2) ? nextPow2((n + 1)/ 2) : maxtn;
	bn =(n + (tn*2 - 1)) / (tn*2);
	if (bn > maxbn) bn = maxbn;

	if (bn > prop.maxGridSize[0] || tn > prop.maxThreadsPerBlock) {
		printf("Dimension error\n");
		exit(-1);
	}
}

template <class T>
void reduce_d(int bn, int tn, int n, T *in, T *out);

template <class T>
bool reduce()
{
	printf("\nReduce: \n");

	bool cpuFinalReduction = true;
	bool needReadBack = true;
	T gpu_result = 0;
	int N = 16*(1<<20);
	int bn, tn;
	gettnbn(bn, tn, N);
	T *in, *out, *d_in, *d_out;
	in = (T*)malloc(N*sizeof(T));
	out = (T*)malloc(bn*sizeof(T));

	cudaMalloc(&d_in, N*sizeof(T)); 
	cudaMalloc(&d_out, bn*sizeof(T));

	if (typeid(T) == typeid(int))
		for (int i = 0; i < N; i++)
			in[i] = (T)(rand() & 0xFF);
	else
		for (int i = 0; i < N; i++)
			in[i] = (rand() & 0xFF) / (T)RAND_MAX;

	cudaMemcpy(d_in, in, N*sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, out, bn*sizeof(T), cudaMemcpyHostToDevice);

	cuda::Timer timer;
	LARGE_INTEGER frequency, t1, t2;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&t1);
	timer.run();
	reduce_d<T>(bn, tn, N, d_in, d_out);
	if (cpuFinalReduction) {
		cuda::SafeCall(cudaMemcpy(out, d_out, bn*sizeof(T), cudaMemcpyDeviceToHost));
		for (int i=0; i<bn; i++)
			gpu_result += out[i];
		needReadBack = false;
	} else {
		int s=bn;
		while (s > 1) {
			gettnbn(bn, tn, s);
			reduce_d<T>(bn, tn, s, d_out, d_out);
			s = (s + (tn*2-1)) / (tn*2);
		}
		if (s > 1) {
			cuda::SafeCall(cudaMemcpy(out, d_out, s*sizeof(T), cudaMemcpyDeviceToHost));
			for (int i=0; i < s; i++)
				gpu_result += out[i];
			needReadBack = false;
		}
	}
	if (needReadBack)
		cuda::SafeCall(cudaMemcpy(&gpu_result, d_out, sizeof(T), cudaMemcpyDeviceToHost));

	timer.stop();
	timer.print();

	cudaDeviceSynchronize();
	QueryPerformanceCounter(&t2);
	printf("Time: %.3f ms.\n", (double)(t2.QuadPart - t1.QuadPart)*1000/frequency.QuadPart);

	T cpu_result = reduceCPU<T>(in, N);

	if (typeid(T) == typeid(int)) {
		printf("\nGPU result = %d\n", gpu_result);
		printf("CPU result = %d\n\n", cpu_result);
	} else {
		printf("\nGPU result = %f\n", gpu_result);
		printf("CPU result = %f\n\n", cpu_result);
	}

	free(in); free(out);
	cudaFree(d_in); cudaFree(d_out);

	return true;
}

}

#endif