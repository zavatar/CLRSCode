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

//////////////////////////////////////////////////////////////////////////
//
// Sort:
// STL(algorithm): sort, stable_sort, partial_sort, partial_sort_copy,
//                 is_sorted, is_sorted_until, nth_element.
//
//////////////////////////////////////////////////////////////////////////
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

//////////////////////////////////////////////////////////////////////////
//
// Merge:
// STL(algorithm): merge, inplace_merge, [includes?], set_union, 
//                 set_intersection, set_difference, set_symmetric_difference
//
//////////////////////////////////////////////////////////////////////////
// NOTE: 
// 1. The infinite number using FLT_MAX violate typename T. updated by Exercise 2.3-2
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

//////////////////////////////////////////////////////////////////////////
//
// Binary Search:
// STL(algorithm): binary_search, equal_range, lower_bound/upper_bound.
//
//////////////////////////////////////////////////////////////////////////
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

//////////////////////////////////////////////////////////////////////////
//
// Partitions:
// STL(algorithm): is_partitioned, partition, stable_partition, 
//                 partition_copy, partition_point
//
//////////////////////////////////////////////////////////////////////////
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

//////////////////////////////////////////////////////////////////////////
//
// Heap:
// STL(algorithm): is_heap, is_heap_until(c11), make_heap, push_heap,
//                 pop_heap, sort_heap.
//
//////////////////////////////////////////////////////////////////////////

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

// O(lgn)
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

// std::make_heap O(3*n)
template <typename T>
void BUILD_MAX_HEAP(T *A, int A_heap_size)
{
	for (int i(A_heap_size/2-1); i >= 0; i--)
		MAX_HEAPIFY(A, A_heap_size, i);
}

// O(nlgn)
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

// O(nlgn)
template <typename T>
void STL_heap_sort(T l, T r)
{
	std::make_heap(l, r); //O(3*n)
	std::sort_heap(l, r); //O(nlgn)
}

//////////////////////////////////////////////////////////////////////////
//
// priority_queue:
// STL(queue): top, push, pop, 
//
//////////////////////////////////////////////////////////////////////////

// priority::top
template <typename T>
inline int HEAP_MAXIMUM(T *A)
{
	return A[0];
}

// pop_heap, priority_queue::pop, O(lgn)
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

// O(lgn)
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

// push_heap, priority_queue::push, O(lgn)
template <typename T>
void MAX_HEAP_INSERT(T *A, int &A_heap_size, T key)
{
	A_heap_size = A_heap_size+1;
	A[A_heap_size] = key;
	HEAP_INCREASE_KEY(A, A_heap_size, key);
}

//////////////////////////////////////////////////////////////////////////
//
//
//
//////////////////////////////////////////////////////////////////////////
template <typename T>
void COUNTING_SORT(T *A, T *B, int A_length, int k)
{
	int *C = new int[k+1];
	memset(C, 0, sizeof(int)*(k+1));//for (int i(0); i <= k; i++) C[i] = 0;
	for (int i(0); i < A_length; i++) C[A[i]]++;
	for (int i(1); i <= k; i++) C[i] += C[i-1];
	if (A == B) {
		T *D = new T[A_length];
		memcpy(D, A, A_length*sizeof(T));
		for (int i(A_length-1); i >= 0; i--) B[--C[D[i]]] = D[i];
		delete []D;
	} else
		for (int i(A_length-1); i >= 0; i--) B[--C[A[i]]] = A[i];
	delete []C;
}

//////////////////////////////////////////////////////////////////////////
//
// nth_element.
//
//////////////////////////////////////////////////////////////////////////
template <typename T>
void BFPRT(T *A, int p, int r, int nth)
{
#define G 5
	if (r-p >= 16) {
		int m=G/2, k=p;
		for (int i=p; i<=r; i+=G)
		{
			if (r-i<G) m=(r-i)/2;
			INSERTION_SORT(A+i, G);
			exchange(A[k++], A[i+m]);
		}
		int mid = (p + k) / 2;
		BFPRT(A, p, k, mid);
		exchange(A[mid], A[r]);
		mid = PARTITION(A, p, r);
		if (mid <= nth)
			BFPRT(A, mid, r, nth);
		else
			BFPRT(A, p, mid, nth);
	} else
		INSERTION_SORT(A+p, r-p+1);
#undef G
}

//////////////////////////////////////////////////////////////////////////
//
// Exercises:
//
//////////////////////////////////////////////////////////////////////////
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

//////////////////////////////////////////////////////////////////////////
//
// Dynamic Programming (DP)
//
//////////////////////////////////////////////////////////////////////////
template <typename T>
void Bottom_Up_Cut_Rod(T* p, int n, T* r, int* s)
{
	T q;
	int i,j;
	r[0]=s[0]=0;
	for (j=0; j<n; r[++j]=q)
		for (i=0, q=-1; i<=j; i++)
			if (q < p[i]+r[j-i]) {
				q = p[i]+r[j-i];
				s[j+1]=i+1; }
}
template <typename T>
void Print_Cut_Rod(T* p, int n)
{
	T *r = new T[n+1];
	int *s = new int[n+1];
	Bottom_Up_Cut_Rod(p, n, r, s);
	while (n>0) {
		std::cout<<s[n]<<" ";
		n-=s[n];
	}
	std::cout<<std::endl;
	delete []r;
	delete []s;
}

// Longest Increase Sequence O(n^2)
template <typename T>
void LIS(T* p, int n, int* f, int* s)
{
	int i,j,q;
	s[0]=-1;
	for (j=0; j<n; f[j++]=q+1)
		for (i=0, q=0; i<j; i++)
			if (p[i]<p[j] && q<f[i])
				q=f[s[j]=i];
}
template <typename T>
int bsearch(T *d, int n, int a)   
{   
	int  l=0, r=n-1;   
	while( l <= r )  
	{   
		int  mid = (l+r)>>1;   
		if( a > d[mid-1] && a <= d[mid] )   
			return mid;             // >&&<= »»Îª: >= && <   
		else if( a <d[mid] )   
			r = mid-1;   
		else l = mid+1;   
	}   
}   
template <typename T>
int fastLIS(T* p, int n, int* f)
{
	T* d = new T[n];
	int i,j,l=1;
	d[0]=p[0]; f[0]=1;
	for (i=1; i<n; i++)
	{
		if (p[i]<=d[0]) j=0;			// <= »»Îª: < 
		else if (p[i]>d[l-1]) j=l++;// > »»Îª: >=
		else j=bsearch(d, l, p[i]);
		d[j]=p[i];
		f[i]=j+1;
	}
	delete []d;
	return l;
}
// Longest Increase Sequence
template <typename T>
int Print_LIS(T* p, int n)
{
	int *f = new int[n];
	int *s = new int[n];
	LIS(p, n, f, s);
	int l=1, k;
	for (int i=0; i<n; i++)
		if (l<f[i]) l=f[k=i];
	for (int i=0; i<n; i++)
		printf("%d ",f[i]);
	printf("\n");
	if (l == fastLIS(p,n,f))
		std::cout<<l<<std::endl;
	for (int i=0; i<n; i++)
		printf("%d ",f[i]);
	printf("\n");
	delete []f;
	for (; k>=0; k=s[k])
		std::cout<<p[k]<<" ";
	std::cout<<std::endl;
	delete []s;
	return l;
}

} // namespace

#endif