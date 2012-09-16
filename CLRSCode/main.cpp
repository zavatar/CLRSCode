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

	//zm::INSERTION_SORT_Benchmark<ISPRINT>(LENGTH);

	zm::MERGE_SORT_Benchmark<ISPRINT>(LENGTH);

	zm::QUICKSORT_Benchmark<ISPRINT, 1>(LENGTH);
	zm::QUICKSORT_Benchmark<ISPRINT, 2>(LENGTH);
	zm::QUICKSORT_Benchmark<ISPRINT, 3>(LENGTH);

	zm::HEAPSORT_Benchmark<ISPRINT>(LENGTH);
	zm::STL_sort_heap_Benchmark<ISPRINT>(LENGTH);

	zm::STL_sort_Benchmark<ISPRINT>(LENGTH);

	zm::BINARY_SEARCH_Benchmark<1>(LENGTH);
	zm::BINARY_SEARCH_Benchmark<2>(LENGTH);

	zm::Exercise_3_3_7<1>(LENGTH);
	zm::Exercise_3_3_7<2>(LENGTH);

	return 0;
}