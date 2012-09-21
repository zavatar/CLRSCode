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

#include "Benchmarks.h"

int main ()
{
	const bool ISPRINT(false);
	const int LENGTH(1<<20); // (1<<20) 1MB, (1<<16) 64KB
	const float CNT(-.5f);

// 	clrs::INSERTION_SORT_Benchmark<ISPRINT>(LENGTH);
// 	pp::isort_Benchmark<ISPRINT, 3>(LENGTH);

// 	clrs::MERGE_SORT_Benchmark<ISPRINT>(LENGTH);
// 
// 	clrs::QUICKSORT_Benchmark<ISPRINT, 1>(LENGTH);
// 	clrs::QUICKSORT_Benchmark<ISPRINT, 2>(LENGTH);
// 	clrs::QUICKSORT_Benchmark<ISPRINT, 3>(LENGTH);
// 
// 	clrs::HEAPSORT_Benchmark<ISPRINT>(LENGTH);
// 	clrs::STL_sort_heap_Benchmark<ISPRINT>(LENGTH);
// 
// 	clrs::STL_sort_Benchmark<ISPRINT>(LENGTH);
// 
// 	clrs::BINARY_SEARCH_Benchmark<1>(LENGTH);
// 	clrs::BINARY_SEARCH_Benchmark<2>(LENGTH);
// 
// 	clrs::Exercise_3_3_7<1>(LENGTH);
// 	clrs::Exercise_3_3_7<2>(LENGTH);

// 	clrs::FIND_MAXIMUM_SUBARRAY_Benchmark<ISPRINT>(LENGTH, CNT);
// 	clrs::Exercise_4_1_4<ISPRINT>(LENGTH, CNT);
// 	clrs::Exercise_4_1_5<ISPRINT>(LENGTH, CNT);
// 	pp::maxsum_Benchmark<ISPRINT, 3>(LENGTH, CNT);
// 	pp::maxsum_Benchmark<ISPRINT, 4>(LENGTH, CNT);

// 	pp::Problems9_5_4<ISPRINT, 1>(LENGTH);
// 	pp::Problems9_5_4<ISPRINT, 2>(LENGTH);

// Temp

	memset_fill_Benchmark(LENGTH);

	return 0;
}