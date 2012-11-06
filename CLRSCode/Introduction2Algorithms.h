//////////////////////////////////////////////////////////////////////////
//
//	Introduction2Algorithms.h
//
//	Written by Meng Zhu ,Zhejiang University (zhumeng1989@gmail.com)
//	Creation date: Sep.10 2012, 19:15
//
//	All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////

#ifndef Introduction2Algorithms_H
#define Introduction2Algorithms_H

#include <cfloat>
#include <algorithm>

namespace clrs {

template <typename T> inline
void exchange(T &a, T &b)
{
	T tmp = a; a = b; b = tmp;
}

// NOTE: 
// 1. For supporting index 0, 'i' must be signed.
template <typename T>
void INSERTION_SORT(T *A, int A_length)
{
	for (int j(1); j < A_length; j++)
	{
		T key = A[j];
		int i = j-1;
		while (i >= 0 && A[i] > key) // $1
		{
			A[i+1] = A[i];
			i = i-1; // $1
		}
		A[i+1] = key;
	}
}

// NOTE: 
// 1. The infinite number using FLT_MAX violate typename T. updated by Excercise 2.3-2
// 2. For indexing from 0, note the change $2.
template <typename T>
void MERGE(T *A, int p, int q, int r)
{
	int n1 = q-p+1;
	int n2 = r-q;
	T *L = new T[n1];
	T *R = new T[n2];
	for (int i(0); i < n1; i++)
		L[i] = A[p+i]; // $2
	for (int j(0); j < n2; j++)
		R[j] = A[q+j+1]; // $2
	int i(0), j(0), k(p);
	for (; i < n1 && j < n2; k++)
	{
		if (L[i] <= R[j])
		{
			A[k] = L[i];
			i = i+1;
		} else {
			A[k] = R[j];
			j = j+1;
		}
	}
	for (; i < n1; i++,k++)
		A[k] = L[i];
	for (; j < n2; j++,k++)
		A[k] = R[j];
	delete []L;
	delete []R;
}

template <typename T>
void MERGE_SORT(T *A, int p, int r)
{
	if (p < r)
	{
		int q = (p+r)/2;
		MERGE_SORT(A, p, q);
		MERGE_SORT(A, q+1, r);
		MERGE(A, p, q, r);
	}
}

// NOTE: 
//
template <typename T>
int ITERATIVE_BINARY_SEARCH(T *A, T v, int low, int high)
{
	while (low <= high)
	{
		int mid = (low+high)/2;
		if (v == A[mid])
			return mid;
		else if (v > A[mid])
			low = mid+1;
		else
			high = mid-1;
	}
	return -1;
}

template <typename T>
int RECURSIVE_BINARY_SEARCH(T *A, T v, int low, int high)
{
	if (low > high)
		return -1;
	int mid = (low+high)/2;
	if (v == A[mid])
		return mid;
	else if (v > A[mid])
		return RECURSIVE_BINARY_SEARCH(A, v, mid+1, high);
	else
		return RECURSIVE_BINARY_SEARCH(A, v, low, mid-1);
}

template <typename T>
int PARTITION(T *A, int p, int r)
{
	T x = A[r];
	int i = p-1;
	for (int j(p); j < r; j++)
	{
		if (A[j] <= x)
		{
			i = i+1;
			exchange(A[i], A[j]);
		}
	}
	exchange(A[i+1], A[r]);
	return i+1;
}

template <typename T>
int RANDOMIZED_PARTITION(T *A, int p, int r)
{
	int i = p+int((float)rand()/(float)RAND_MAX*(r-p));
	exchange(A[i], A[r]);
	return PARTITION(A, p, r);
}

template <typename T>
void QUICKSORT(T *A, int p, int r)
{
	if (p < r)
	{
		int q = PARTITION(A, p, r);
		QUICKSORT(A, p, q-1);
		QUICKSORT(A, q+1, r);
	}
}

template <typename T>
void RANDOMIZED_QUICKSORT(T *A, int p, int r)
{
	if (p < r)
	{
		int q = RANDOMIZED_PARTITION(A, p, r);
		RANDOMIZED_QUICKSORT(A, p, q-1);
		RANDOMIZED_QUICKSORT(A, q+1, r);
	}
}

template <typename T>
void TAIL_RECURSIVE_QUICKSORT(T *A, int p, int r)
{
	while (p < r)
	{
		int q = PARTITION(A, p, r);
		TAIL_RECURSIVE_QUICKSORT(A, p, q-1);
		p = q+1;
	}
}

// NOTE: 
// 1. The infinite number using FLT_MAX violate typename T. 
// (Could be solved using template in the future.)
template <typename T>
T FIND_MAX_CROSSING_SUBARRAY(T *A, int low, int mid, int high,
	int &max_left, int &max_right)
{
	T left_sum = -FLT_MAX; // $1
	T sum = 0;
	for (int i(mid); i >= low; i--)
	{
		sum = sum + A[i];
		if (sum > left_sum)
		{
			left_sum = sum;
			max_left = i;
		}
	}
	T right_sum = -FLT_MAX; // $1
	sum = 0;
	for (int j(mid+1); j <= high; j++)
	{
		sum = sum + A[j];
		if (sum > right_sum)
		{
			right_sum = sum;
			max_right = j;
		}
	}
	return left_sum + right_sum;
}

template <typename T>
T FIND_MAXIMUM_SUBARRAY(T *A, int low, int high,
	int &max_left, int &max_right)
{
	if (high == low)
	{
		max_left = low;
		max_right = high;
		return A[low];
	}

	int mid = (low + high) / 2;
	int left_low, left_high;
	T left_sum = FIND_MAXIMUM_SUBARRAY(A, low, mid, left_low, left_high);
	int right_low, right_high;
	T right_sum = FIND_MAXIMUM_SUBARRAY(A, mid+1, high, right_low, right_high);
	int cross_low, cross_high;
	T cross_sum = FIND_MAX_CROSSING_SUBARRAY(A, low, mid, high, cross_low, cross_high);

	if (left_sum >= right_sum && left_sum >= cross_sum)
	{
		max_left = left_low;
		max_right = left_high;
		return left_sum;
	} else if (right_sum >= left_sum && right_sum >= cross_sum)
	{
		max_left = right_low;
		max_right = right_high;
		return right_sum;
	} else {
		max_left = cross_low;
		max_right = cross_high;
		return cross_sum;
	}
}

inline int HEAP_PARENT(int i)
{
	return i/2;
}

inline int HEAP_LEFT(int i)
{
	return 2*i;
}

inline int HEAP_RIGHT(int i)
{
	return 2*i+1;
}

template <typename T>
void MAX_HEAPIFY(T *A, int A_heap_size, int i)
{
	int l = HEAP_LEFT(i);
	int r = HEAP_RIGHT(i);
	int largest;
	if (l < A_heap_size && A[l] > A[i])
		largest = l;
	else
		largest = i;
	if (r < A_heap_size && A[r] > A[largest])
		largest = r;
	if (largest != i)
	{
		exchange(A[i], A[largest]);
		MAX_HEAPIFY(A, A_heap_size, largest);
	}
}

template <typename T>
void BUILD_MAX_HEAP(T *A, int A_heap_size)
{
	for (int i(A_heap_size/2-1); i >= 0; i--)
		MAX_HEAPIFY(A, A_heap_size, i);
}

template <typename T>
void STL_heap_sort(T l, T r)
{
	std::make_heap(l, r);
	std::sort_heap(l, r);
}

template <typename T>
void HEAPSORT(T *A, int A_length)
{
	int A_heap_size = A_length;
	BUILD_MAX_HEAP(A, A_heap_size);
	for (int i(A_length-1); i > 0; i--)
	{
		exchange(A[0], A[i]);
		A_heap_size = A_heap_size-1;
		MAX_HEAPIFY(A, A_heap_size, 0);
	}
}

template <typename T>
inline int HEAP_MAXIMUM(T *A)
{
	return A[0];
}

template <typename T>
int HEAP_EXTRACT_MAX(T *A, int &A_heap_size)
{
	if (A_heap_size < 1) {
		printf("heap underflow\n");
		exit(-1);
	}
	int max = A[0];
	A[0] = A[A_heap_size-1];
	A_heap_size = A_heap_size-1;
	MAX_HEAPIFY(A, A_heap_size, 0);
	return max;
}

template <typename T>
void HEAP_INCREASE_KEY(T *A, int i, T key)
{
	if (key < A[i]) {
		printf("new key is smaller than current key\n");
		exit(-1);
	}
	A[i] = key;
	while (i > 0 && A[HEAP_PARENT(i)] < A[i])
	{
		exchange(A[i], A[HEAP_PARENT(i)]);
		i = HEAP_PARENT(i);
	}
}

template <typename T>
void MAX_HEAP_INSERT(T *A, int &A_heap_size, T key)
{
	A_heap_size = A_heap_size+1;
	A[A_heap_size] = key;
	HEAP_INCREASE_KEY(A, A_heap_size, key);
}

} // namespace

#endif