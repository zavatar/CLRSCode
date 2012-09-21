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
#include "ProgrammingPearls.h"

#include <iostream>
#include <ctime>
#include <vector>
#include <algorithm>
#include <windows.h>

namespace {

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
void ARRAYADD(float *A, int A_length, float add)
{
	for (int i(0); i < A_length; i++)
		A[i] += add;
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

} // namespace

namespace clrs {

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
	QUICKSORT(A, 0, LENGTH-1);

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
void FIND_MAXIMUM_SUBARRAY_Benchmark(int LENGTH, float cnt)
{
	LARGE_INTEGER t1, t2;
	std::cout<<"\nFIND_MAXIMUM_SUBARRAY: \n"; 

	float *A = new float[LENGTH];
	RANDOMFILL<false>(A, LENGTH);
	ARRAYADD<ISPRINT>(A, LENGTH, cnt);

	int left, right;
	QueryPerformanceCounter(&t1);

	float sum = FIND_MAXIMUM_SUBARRAY(A, 0, LENGTH-1, left, right);

	QueryPerformanceCounter(&t2); ELAPSEDTIME(t1, t2);
	std::cout << "Maxsum: " << sum << ", in (" << left << "," << right << ")"<< std::endl;

	delete []A;
}

template <bool ISPRINT>
void Exercise_4_1_4(int LENGTH, float cnt)
{
	std::cout<<"\nExercise 4.1-4:"; 
	pp::maxsum_Benchmark<ISPRINT, 3>(LENGTH, cnt);
}

template <bool ISPRINT>
void Exercise_4_1_5(int LENGTH, float cnt)
{
	std::cout<<"\nExercise 4.1-5:"; 
	pp::maxsum_Benchmark<ISPRINT, 4>(LENGTH, cnt);
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

namespace pp {

template <bool ISPRINT>
void maxsum3_Benchmark(int LENGTH, float cnt)
{
	LARGE_INTEGER t1, t2;
	std::cout<<"\nmaxsum3: \n"; 

	float *A = new float[LENGTH];
	RANDOMFILL<false>(A, LENGTH);
	ARRAYADD<ISPRINT>(A, LENGTH, cnt);

	int left, right;
	QueryPerformanceCounter(&t1);

	float sum = maxsum3(A, LENGTH, left, right);

	QueryPerformanceCounter(&t2); ELAPSEDTIME(t1, t2);
	std::cout << "Maxsum: " << sum << ", in (" << left << "," << right << ")"<< std::endl;

	delete []A;
}

template <bool ISPRINT, int WHICH>
void maxsum_Benchmark(int LENGTH, float cnt)
{
	LARGE_INTEGER t1, t2;
	std::cout<<"\nmaxsum: (" << WHICH << ")\n"; 

	float *A = new float[LENGTH];
	RANDOMFILL<false>(A, LENGTH);
	ARRAYADD<ISPRINT>(A, LENGTH, cnt);

	int left, right;
	QueryPerformanceCounter(&t1);

	float sum;
	switch (WHICH)
	{
	case 3:
		sum = maxsum3(A, 0, LENGTH-1, left, right);
		break;
	case 4:
		sum = maxsum4(A, LENGTH, left, right);
		break;
	default:
		sum = 0;
		std::cout << "Has not been Realized...";
	}

	QueryPerformanceCounter(&t2); ELAPSEDTIME(t1, t2);
	std::cout << "Maxsum: " << sum << ", in (" << left << "," << right << ")"<< std::endl;

	delete []A;
}

template <bool ISPRINT, int WHICH>
void isort_Benchmark(int LENGTH)
{
	LARGE_INTEGER t1, t2;
	std::cout<<"\nisort: (" << WHICH << ")\n"; 

	float *A = new float[LENGTH];
	RANDOMFILL<ISPRINT>(A, LENGTH);

	QueryPerformanceCounter(&t1);

	switch (WHICH)
	{
	case 1:
		isort1(A, LENGTH);
		break;
	case 2:
		isort2(A, LENGTH);
		break;
	case 3:
		isort3(A, LENGTH);
		break;
	default:
		std::cout << "Has not been Realized...";
	}

	QueryPerformanceCounter(&t2); ELAPSEDTIME(t1, t2);

	ASSERTSORTED<ISPRINT>(A, LENGTH);
	delete []A;
}

template <bool ISPRINT, int WHICH>
void Problems9_5_4(int LENGTH)
{
	LARGE_INTEGER t1, t2;
	std::cout<<"\nProblems9_5_4: (" << WHICH << ")\n";

	float *A = new float[LENGTH];
	RANDOMFILL<ISPRINT>(A, LENGTH);

	QueryPerformanceCounter(&t1);

	float maximum;
	switch (WHICH)
	{
	case 1:
		maximum = arrmax(A, LENGTH);
		break;
	case 2:
		maximum = arrmaxMacro(A, LENGTH);
		break;
	default:
		maximum = 0;
		std::cout << "Has not been Realized...";
	}

	QueryPerformanceCounter(&t2); ELAPSEDTIME(t1, t2);

	switch (WHICH)
	{
	case 1:
		std::cout << "Inline Maximum: " << maximum << std::endl;
		break;
	case 2:
		std::cout << "Macro Maximum: " << maximum << std::endl;
		break;
	}

	delete []A;
}

} // namespace

// Temp
void memset_fill_Benchmark(int LENGTH)
{
	LARGE_INTEGER t1, t2;
	std::cout<<"\nSTL_fill: \n";

	std::vector<float> vA(LENGTH);

	QueryPerformanceCounter(&t1);

	std::fill(vA.begin(), vA.end(), 1.f);

	QueryPerformanceCounter(&t2); ELAPSEDTIME(t1, t2);

	std::cout<<"\nmemset: \n";

	QueryPerformanceCounter(&t1);

	memset(&vA[0], 0, vA.size()*sizeof(float));

	QueryPerformanceCounter(&t2); ELAPSEDTIME(t1, t2);

}

#endif