//////////////////////////////////////////////////////////////////////////
//
//	Benchmarks.h
//
//	Written by Meng Zhu ,Zhejiang University (zhumeng1989@gmail.com)
//	Creation date: Sep.15 2012, 21:26
//
//	All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////

#ifndef Benchmarks_H
#define Benchmarks_H

#include "Introduction2Algorithms.h"
#include "Exercises.h"

#include <iostream>
#include <ctime>
#include <vector>
#include <algorithm>
#include <windows.h>

namespace zm {

void PRINT(float *A, int A_length)
{
	for (int i(0); i < A_length; i++)
		std::cout << A[i] << ' ';
	std::cout << std::endl;
}

void ELAPSEDTIME(LARGE_INTEGER t1, LARGE_INTEGER t2)
{
	LARGE_INTEGER frequency;
	double elapsedTime;

	QueryPerformanceFrequency(&frequency);

	elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
	std::cout << elapsedTime << " ms.\n";
}

template <bool ISPRINT>
void RANDOMFILL(float *A, int A_length)
{
	srand( (unsigned int)time(NULL) );
	for (int i(0); i < A_length; i++)
		A[i] = (float)rand()/(float)RAND_MAX;
	if (ISPRINT)
		PRINT(A, A_length);
}

template <bool ISPRINT>
void ASSERTSORTED(float *A, int A_length)
{
	for (int i(0); i < A_length-1; i++)
		if (A[i] > A[i+1]) {
			printf("Unsorted\n");
			exit(-1);
		}
	if (ISPRINT)
		PRINT(A, A_length);
}

template <bool ISPRINT>
void INSERTION_SORT_Benchmark(int LENGTH)
{
	LARGE_INTEGER t1, t2;
	std::cout<<"\nINSERTION_SORT: \n"; 

	float *A = new float[LENGTH];
	RANDOMFILL<ISPRINT>(A, LENGTH);

	QueryPerformanceCounter(&t1);

	INSERTION_SORT(A, LENGTH);

	QueryPerformanceCounter(&t2); ELAPSEDTIME(t1, t2);

	ASSERTSORTED<ISPRINT>(A, LENGTH);
	delete []A;
}

template <bool ISPRINT>
void MERGE_SORT_Benchmark(int LENGTH)
{
	LARGE_INTEGER t1, t2;
	std::cout<<"\nMERGE_SORT: \n";

	float *A = new float[LENGTH];
	RANDOMFILL<ISPRINT>(A, LENGTH);

	QueryPerformanceCounter(&t1);

	MERGE_SORT(A, 0, LENGTH-1);

	QueryPerformanceCounter(&t2); ELAPSEDTIME(t1, t2);

	ASSERTSORTED<ISPRINT>(A, LENGTH);
	delete []A;
}

template <bool ISPRINT, int WHICH>
void QUICKSORT_Benchmark(int LENGTH)
{
	LARGE_INTEGER t1, t2;
	std::cout<<"\nQUICKSORT (" << WHICH << "): \n"; 

	float *A = new float[LENGTH];
	RANDOMFILL<ISPRINT>(A, LENGTH);

	QueryPerformanceCounter(&t1);

	// You could try three QuickSort codes
	switch (WHICH)
	{
	case 2:
		RANDOMIZED_QUICKSORT(A, 0, LENGTH-1);
		break;
	case 3:
		TAIL_RECURSIVE_QUICKSORT(A, 0, LENGTH-1);
		break;
	case 1:
	default:
		QUICKSORT(A, 0, LENGTH-1);
	}

	QueryPerformanceCounter(&t2); ELAPSEDTIME(t1, t2);

	ASSERTSORTED<ISPRINT>(A, LENGTH);
	delete []A;
}

template <bool ISPRINT>
void STL_sort_Benchmark(int LENGTH)
{
	LARGE_INTEGER t1, t2;
	std::cout<<"\nSTL_sort: \n";

	std::vector<float> vA(LENGTH);
	RANDOMFILL<ISPRINT>(&vA[0], LENGTH);

	QueryPerformanceCounter(&t1);

	sort(vA.begin(), vA.end());

	QueryPerformanceCounter(&t2); ELAPSEDTIME(t1, t2);

	ASSERTSORTED<ISPRINT>(&vA[0], LENGTH);
}

template <int WHICH>
void BINARY_SEARCH_Benchmark(int LENGTH)
{
	LARGE_INTEGER t1, t2;
	std::cout << "\nBINARY_SEARCH:  (" << WHICH << ")\n";

	float *A = new float[LENGTH];
	RANDOMFILL<false>(A, LENGTH);

	// Remember sort first
	zm::QUICKSORT(A, 0, LENGTH-1);

	int Idx = int((float)rand()/(float)RAND_MAX*(LENGTH-1));
	std::cout << "Idx = " << Idx << std::endl;
	QueryPerformanceCounter(&t1);

	int idx;
	switch (WHICH)
	{
	case 2:
		idx = RECURSIVE_BINARY_SEARCH(A, A[Idx], 0, LENGTH-1);
		break;
	case 1:
	default:
		idx = ITERATIVE_BINARY_SEARCH(A, A[Idx], 0, LENGTH-1);
	}
	

	QueryPerformanceCounter(&t2); ELAPSEDTIME(t1, t2);
	std::cout << A[Idx] << "==" << A[idx] << std::endl;

	delete []A;
}

template <int WHICH>
void Exercise_3_3_7(int LENGTH)
{
	LARGE_INTEGER t1, t2;
	std::cout<<"\nExercise 2.3-7:  (" << WHICH << ")\n"; 

	float *A = new float[LENGTH];
	RANDOMFILL<false>(A, LENGTH);

	int Idx1 = int((float)rand()/(float)RAND_MAX*(LENGTH-1));
	int Idx2 = int((float)rand()/(float)RAND_MAX*(LENGTH-1));
	// if Idx1 == Idx2, then it may print No.
	std::cout << "Pair: <" << Idx1 << ',' << Idx2 << ">" << std::endl;

	// you can set sum1 = -1 to compare (1) and (2)
	float sum1 = A[Idx1]+A[Idx2];
	float sum2;
	QueryPerformanceCounter(&t1);

	switch (WHICH)
	{
	case 2:
		sum2 = sum_exist2(A, sum1, LENGTH);
		break;
	case 1:
	default:
		sum2 = sum_exist1(A, sum1, LENGTH);
	}

	QueryPerformanceCounter(&t2); ELAPSEDTIME(t1, t2);
	std::cout << sum1 << "==" << sum2 << std::endl;

	delete []A;
}

template <bool ISPRINT>
void HEAPSORT_Benchmark(int LENGTH)
{
	LARGE_INTEGER t1, t2;
	std::cout<<"\nHEAPSORT: \n";

	float *A = new float[LENGTH];
	RANDOMFILL<ISPRINT>(A, LENGTH);

	QueryPerformanceCounter(&t1);

	HEAPSORT(A, LENGTH);

	QueryPerformanceCounter(&t2); ELAPSEDTIME(t1, t2);

	ASSERTSORTED<ISPRINT>(A, LENGTH);
	delete []A;
}

template <bool ISPRINT>
void STL_sort_heap_Benchmark(int LENGTH)
{
	LARGE_INTEGER t1, t2;
	std::cout<<"\nSTL_sort_heap: \n";

	std::vector<float> vA(LENGTH);
	RANDOMFILL<ISPRINT>(&vA[0], LENGTH);

	QueryPerformanceCounter(&t1);

	std::make_heap(vA.begin(), vA.end());
	std::sort_heap(vA.begin(), vA.end());

	QueryPerformanceCounter(&t2); ELAPSEDTIME(t1, t2);

	ASSERTSORTED<ISPRINT>(&vA[0], LENGTH);
}

} // namespace

#endif