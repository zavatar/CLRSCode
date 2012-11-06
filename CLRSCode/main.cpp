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

#include <thrust/version.h>

#include <ctime>
#include <vector>

void cudaMain();

int main ()
{

	cuda::InitDev();

	int major = THRUST_MAJOR_VERSION;
	int minor = THRUST_MINOR_VERSION;

	std::cout << "Thrust v" << major << "." << minor << std::endl;

	typedef bm::Type Type;

	bm::BenchMarks<Type, bm::LENGTH, bm::ISPRINT> benchmarks;

// 	benchmarks.sort(clrs::INSERTION_SORT<Type>, "\nINSERTION_SORT: \n");
// 
// 	benchmarks.sort(pp::isort3<Type>, "\nisort3: \n");
// 
// 	benchmarks.sort(clrs::MERGE_SORT<Type>, "\nMERGE_SORT: \n");
// 
 	benchmarks.sort(std::sort<Type*>, "\nSTL_sort: \n");
// 
// 	benchmarks.sort(clrs::QUICKSORT<Type>, "\nQUICKSORT : \n");
// 
// 	benchmarks.sort(clrs::HEAPSORT<Type>, "\nHEAPSORT: \n");
// 	
// 	benchmarks.sort(clrs::STL_heap_sort<Type*>, "\nSTL_sort_heap: \n");
//
//	benchmarks.search(clrs::ITERATIVE_BINARY_SEARCH<Type>, "\nBINARY_SEARCH: \n");
//
	cudaMain();
// 
// 	benchmarks.sum_exist(clrs::Exercise_3_3_7<Type>, "\nExercise 2.3-7: \n");
// 
// 	benchmarks.maxsum(clrs::FIND_MAXIMUM_SUBARRAY<Type>, bm::CNT,
// 		"\nFIND_MAXIMUM_SUBARRAY: \n");
// 
// 	benchmarks.maxsum(pp::maxsum3<Type>, bm::CNT, "\nmaxsum3: \n");
// 	benchmarks.maxsum(clrs::FIND_MAXIMUM_SUBARRAY_DAC<Type>, bm::CNT,
// 		"\nFIND_MAXIMUM_SUBARRAY_DAC: \n");
// 
// 	benchmarks.maxsum(pp::maxsum4<Type>, bm::CNT, "\nmaxsum4: \n");
// 	benchmarks.maxsum(clrs::FIND_MAXIMUM_SUBARRAY_SCAN<Type>, bm::CNT,
// 		"\nFIND_MAXIMUM_SUBARRAY_SCAN: \n");
// 
// 	// pp->Problems9_5_4, suggest LENGTH = 1<<5
// 	benchmarks.arrmax(pp::arrmax<Type>, "\nProblems9_5_4: (Inline)\n");
// 	benchmarks.arrmax(pp::arrmaxMacro<Type>, "\nProblems9_5_4: (Macro)\n");
// 
// 	mc::Typelist_Benchmark();
// 
// 	mc::Functor_Benchmark();

	pf::SAXPY();

	return 0;
}