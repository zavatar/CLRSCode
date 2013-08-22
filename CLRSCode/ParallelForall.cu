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

extern "C"
bool isPow2(unsigned int x);

namespace pf {

__global__
void saxpy_K(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) 
	  y[i] = a*x[i] + y[i];
}

void SAXPY_D(int bn, int tn, int n, float a, float *x, float *y)
{
	saxpy_K<<<bn, tn>>>(n, a, x, y);
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  64];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T *smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = mySum = mySum + smem[tid + 32];
        }

        if (blockSize >=  32)
        {
            smem[tid] = mySum = mySum + smem[tid + 16];
        }

        if (blockSize >=  16)
        {
            smem[tid] = mySum = mySum + smem[tid +  8];
        }

        if (blockSize >=   8)
        {
            smem[tid] = mySum = mySum + smem[tid +  4];
        }

        if (blockSize >=   4)
        {
            smem[tid] = mySum = mySum + smem[tid +  2];
        }

        if (blockSize >=   2)
        {
            smem[tid] = mySum = mySum + smem[tid +  1];
        }
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

template <class T>
void reduce_d(int bn, int tn, int n, T *in, T *out)
{
	dim3 dB(tn, 1, 1);
    dim3 dG(bn, 1, 1);

	int smn = (tn <= 32) ? 2 * tn * sizeof(T) : tn * sizeof(T);

	if (isPow2(n))
    {
        switch (tn)
        {
            case 512:
                reduce6<T, 512, true><<< dG, dB, smn >>>(in, out, n); break;
            case 256:
                reduce6<T, 256, true><<< dG, dB, smn >>>(in, out, n); break;
            case 128:
                reduce6<T, 128, true><<< dG, dB, smn >>>(in, out, n); break;
            case 64:
                reduce6<T,  64, true><<< dG, dB, smn >>>(in, out, n); break;
            case 32:
                reduce6<T,  32, true><<< dG, dB, smn >>>(in, out, n); break;
            case 16:
                reduce6<T,  16, true><<< dG, dB, smn >>>(in, out, n); break;
            case  8:
                reduce6<T,   8, true><<< dG, dB, smn >>>(in, out, n); break;
            case  4:
                reduce6<T,   4, true><<< dG, dB, smn >>>(in, out, n); break;
            case  2:
                reduce6<T,   2, true><<< dG, dB, smn >>>(in, out, n); break;
            case  1:
                reduce6<T,   1, true><<< dG, dB, smn >>>(in, out, n); break;
        }
    }
    else
    {
        switch (tn)
        {
            case 512:
                reduce6<T, 512, false><<< dG, dB, smn >>>(in, out, n); break;
            case 256:
                reduce6<T, 256, false><<< dG, dB, smn >>>(in, out, n); break;
            case 128:
                reduce6<T, 128, false><<< dG, dB, smn >>>(in, out, n); break;
            case 64:
                reduce6<T,  64, false><<< dG, dB, smn >>>(in, out, n); break;
            case 32:
                reduce6<T,  32, false><<< dG, dB, smn >>>(in, out, n); break;
            case 16:
                reduce6<T,  16, false><<< dG, dB, smn >>>(in, out, n); break;
            case  8:
                reduce6<T,   8, false><<< dG, dB, smn >>>(in, out, n); break;
            case  4:
                reduce6<T,   4, false><<< dG, dB, smn >>>(in, out, n); break;
            case  2:
                reduce6<T,   2, false><<< dG, dB, smn >>>(in, out, n); break;
            case  1:
                reduce6<T,   1, false><<< dG, dB, smn >>>(in, out, n); break;
        }
    }
}

template void
reduce_d<float>(int bn, int tn, int n, float *in, float *out);

template void
reduce_d<int>(int bn, int tn, int n, int *in, int *out);

}