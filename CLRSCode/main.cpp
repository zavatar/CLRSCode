//////////////////////////////////////////////////////////////////////////
//
//	Sort.cpp
//
//	Written by Meng Zhu ,Zhejiang University (zhumeng1989@gmail.com)
//	Creation date: Sep.10 2012, 19:15
//
//	All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////

// Macros Setup
//#define MCXX11

#include "Benchmarks.h"

#include "Introduction2Algorithms.h"
#include "ProgrammingPearls.h"
#include "Exercises.h"
#include "ModernC++.h"
#include "ParallelForall.h"
#include "ThrustBenchmarks.h"
#define _BSGP_INITED
#include "BSGP.h"

#include <thrust/version.h>

#include <ctime>
#include <vector>

#define TESTSORT \
	benchmarks.sort(clrs::INSERTION_SORT<Type>, "\nINSERTION_SORT: \n");\
	benchmarks.sort(pp::isort3<Type>, "\nisort3: \n");\
	benchmarks.sort(clrs::MERGE_SORT<Type>, "\nMERGE_SORT: \n");\
	benchmarks.sort(std::sort<Type*>, "\nSTL_sort: \n");\
	benchmarks.sort(clrs::QUICKSORT<Type>, "\nQUICKSORT : \n");\
	benchmarks.sort(clrs::HEAPSORT<Type>, "\nHEAPSORT: \n");\
	benchmarks.sort(clrs::STL_heap_sort<Type*>, "\nSTL_sort_heap: \n");\
	benchmarks.sort(mythrust::sort<Type>, "\nThrust_sort: \n");\

#define MAXSUM \
	benchmarks.maxsum(clrs::FIND_MAXIMUM_SUBARRAY<Type>, bm::CNT,"\nFIND_MAXIMUM_SUBARRAY: \n");\
	benchmarks.maxsum(pp::maxsum3<Type>, bm::CNT, "\nmaxsum3: \n");\
	benchmarks.maxsum(clrs::FIND_MAXIMUM_SUBARRAY_DAC<Type>, bm::CNT,"\nFIND_MAXIMUM_SUBARRAY_DAC: \n");\
	benchmarks.maxsum(pp::maxsum4<Type>, bm::CNT, "\nmaxsum4: \n");\
	benchmarks.maxsum(clrs::FIND_MAXIMUM_SUBARRAY_SCAN<Type>, bm::CNT,"\nFIND_MAXIMUM_SUBARRAY_SCAN: \n");\

void cudaMain();

void producer_consumer();

//////////////////////////////////////////////////////////////////////////
//
// TODO: next_permutation, prev_permutation.
//
//////////////////////////////////////////////////////////////////////////
int main ()
{
	cuda::Device dev;
	dev.PrintProp();

	int major = THRUST_MAJOR_VERSION;
	int minor = THRUST_MINOR_VERSION;

	std::cout << "Thrust v" << major << "." << minor << std::endl;

	typedef unsigned char Type;

	bm::BenchMarks<Type> benchmarks;

	int p[10]={1,5,8,9,10,17,17,20,24,30};
	clrs::Print_Cut_Rod(p, 7);
	clrs::Print_LIS(p, 10);

	//TESTSORT

	//benchmarks.search(clrs::ITERATIVE_BINARY_SEARCH<Type>, "\nBINARY_SEARCH: \n");
	//benchmarks.sum_exist(clrs::Exercise_3_3_7<Type>, "\nExercise 2.3-7: \n");

	//MAXSUM

	// pp->Problems9_5_4, suggest LENGTH = 1<<5
//	benchmarks.arrmax(pp::arrmax<Type>, "\nProblems9_5_4: (Inline)\n");
//	benchmarks.arrmax(pp::arrmaxMacro<Type>, "\nProblems9_5_4: (Macro)\n");

	// suggest LENGTH = 1<<8
	//benchmarks.benchBFPRT();
// 
// 	mc::Typelist_Benchmark();
// 
// 	mc::Functor_Benchmark();

//	pf::SAXPY();

//	producer_consumer();

//	pf::reduce<Type>();
	
//	char A[] = {6,0,1,2,3,6,1,3,6,2};
//	clrs::COUNTING_SORT(A, A, 10, 256);

//	BSGPmain();

	system("pause");

	return 0;
}