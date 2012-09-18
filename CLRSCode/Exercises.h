//////////////////////////////////////////////////////////////////////////
//
//	Exercises.h
//
//	Written by Meng Zhu ,Zhejiang University (zhumeng1989@gmail.com)
//	Creation date: Sep.10 2012, 19:15
//
//	All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////

#ifndef Exercises_H
#define Exercises_H

#include "Introduction2Algorithms.h"

namespace clrs {

// NOTE: Exercise 2.3-7
// Describe a жи(n*lgn)-time algorithm that, given a set S of n integers and
// another integer x, determines whether or not there exist two elements 
// in S whose sum is exactly x.
template <typename T>
T sum_exist1(T *A, T x, int A_length)
{
	int i(0), j(A_length-1);
	QUICKSORT(A, 0, j);
	while (i < j)
	{
		T sum = A[i]+A[j];
		if (sum > x)
			j--;
		else if (sum < x)
			i++;
		else
			break;
	}
	if (i < j) {
		printf("(%d,%d) YES\n", i, j);
		return A[i]+A[j];
	} else {
		printf("NO\n");
		return 0;
	}
}

// NOTE: 
// After sort, do binary search in the array.
template <typename T>
T sum_exist2(T *A, T x, int A_length)
{
	int i(0), r(A_length-1), j;
	QUICKSORT(A, 0, r);
	for (; i < r; i++)
	{
		T x_Ai = x - A[i];
		j = ITERATIVE_BINARY_SEARCH(A, x_Ai, i+1, r);
		if (j != -1)
			break;
	}
	if (i < r) {
		printf("(%d,%d) YES\n", i, j);
		return A[i]+A[j];
	} else {
		printf("NO\n");
		return 0;
	}
}

// NOTE: Exercise 4.1-4, I am not sure.
template <typename T>
T FIND_MAXIMUM_SUBARRAY_DAC(T *A, int low, int high, int &left, int &right)
{
	return pp::maxsum3(A, low, high, left, right);
}

// NOTE: Exercise 4.1-5
template <typename T>
T FIND_MAXIMUM_SUBARRAY_SCAN(T *A, int A_length, int &left, int &right)
{
	return pp::maxsum4(A, A_length, left, right);
}

} // namespace

#endif