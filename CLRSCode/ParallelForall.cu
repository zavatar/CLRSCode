//////////////////////////////////////////////////////////////////////////
//
//	ParallelForall.cu
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

#include <device_launch_parameters.h>

namespace pf {

__global__
void saxpy_K(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

void SAXPY_D(int nb, int nt, int n, float a, float *x, float *y)
{
	saxpy_K<<<nb, nt>>>(n, a, x, y);
}

}