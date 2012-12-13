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

#include <iostream>
#include <algorithm>
#include <windows.h>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace bm {

typedef float Type;

const bool ISPRINT(false);
const int LENGTH(1<<20); // (1<<20) 1MB, (1<<16) 64KB
const Type CNT(-.5f);

template <typename T>
void _sort(void(*fun)(T*, T*), T*A, int N)
{
	fun(A, A+N);
}

template <typename T>
void _sort(void(*fun)(T*, int), T *A, int N)
{
	fun(A, N);
}

template <typename T>
void _sort(void(*fun)(T*, int, int), T *A, int N)
{
	fun(A, 0, N-1);
}

template <typename T>
int _search(int(*fun)(T*, T, int, int), T *A, T v, int N)
{
	return fun(A, v, 0, N-1);
}

template <typename T>
T _maxsum(T(*fun)(T*, int, int, int&, int&), T *A, int N, int &l, int &r)
{
	return fun(A, 0, N-1, l, r);
}

template <typename T>
T _maxsum(T(*fun)(T*, int, int&, int&), T *A, int N, int &l, int &r)
{
	return fun(A, N, l, r);
}

template <typename T, int LENGTH = 1<<20, bool ISPRINT = false>
class BenchMarks {

public:

	BenchMarks() {
		A = new T[LENGTH];
	}

	~BenchMarks() {
		delete []A;
	}

	template <class Fun>
	void sort(Fun fun, const char* cout)
	{
		std::cout << cout;

		randomFill(A, A+LENGTH);

		QueryPerformanceCounter(&t1);

		_sort(fun, A, LENGTH);

		QueryPerformanceCounter(&t2); elapsedTime(t1, t2);

		assertSorted(A, A+LENGTH);
	}

	template <class Fun>
	void search(Fun fun, const char* cout)
	{
		std::cout << cout;

		randomFill(A, A+LENGTH);

		// Remember sort first
		std::sort(A, A+LENGTH);

		int Idx = int(_rand<T>()*(LENGTH-1));
		std::cout << "Idx = " << Idx << std::endl;

		QueryPerformanceCounter(&t1);

		int idx = _search(fun, A, A[Idx], LENGTH);

		QueryPerformanceCounter(&t2); elapsedTime(t1, t2);

		std::cout << A[Idx] << "==" << A[idx] << std::endl;
	}

	template <class Fun>
	void sum_exist(Fun fun, const char* cout)
	{
		std::cout << cout;

		randomFill(A, A+LENGTH);

		int Idx1 = int(_rand<T>()*(LENGTH-1));
		int Idx2 = int(_rand<T>()*(LENGTH-1));
		// if Idx1 == Idx2, then it may print No.
		std::cout << "Pair: <" << Idx1 << ',' << Idx2 << ">" << std::endl;

		// you can set sum1 = -1 to compare (sum_exist1) and (sum_exist2)
		T sum1 = A[Idx1]+A[Idx2];

		QueryPerformanceCounter(&t1);

		T sum2 = fun(A, sum1, LENGTH);

		QueryPerformanceCounter(&t2); elapsedTime(t1, t2);

		std::cout << sum1 << "==" << sum2 << std::endl;
	}

	template <class Fun>
	void maxsum(Fun fun, T cnt, const char* cout)
	{
		std::cout << cout;

		randomFill(A, A+LENGTH);

		arrayAdd(A, A+LENGTH, cnt);

		int left, right;
		QueryPerformanceCounter(&t1);

		T sum = _maxsum(fun, A, LENGTH, left, right);

		QueryPerformanceCounter(&t2); elapsedTime(t1, t2);

		std::cout << "Maxsum: " << sum << ", in (" << left << "," << right << ")"<< std::endl;
	}

	template <class Fun>
	void arrmax(Fun fun, const char* cout)
	{
		std::cout << cout;

		randomFill(A, A+LENGTH);

		QueryPerformanceCounter(&t1);

		T maximum = fun(A, LENGTH);

		QueryPerformanceCounter(&t2); elapsedTime(t1, t2);

		std::cout << "Maximum: " << maximum << std::endl;
	}

private:

	LARGE_INTEGER t1, t2;
	T *A;

	void elapsedTime(LARGE_INTEGER t1, LARGE_INTEGER t2)
	{
		LARGE_INTEGER frequency;
		double elapsedTime;

		QueryPerformanceFrequency(&frequency);

		elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
		std::cout << elapsedTime << " ms.\n";
	}

	void print(T *l, T *r)
	{
		for (int i(0); i < r-l; i++)
			std::cout << l[i] << ' ';
		std::cout << std::endl;
	}

	template <typename T>
	T _rand();

	template <>
	float _rand<float>() {
		return (float)rand()/(float)RAND_MAX;
	}

	template <>
	int _rand<int>() {
		return rand();
	}

	template <>
	char _rand<char>() {
		return char(rand());
	}

	void randomFill(T *l, T *r)
	{
		srand( (unsigned int)time(NULL) );
		for (int i(0); i < r-l; i++)
			l[i] = _rand<T>();
		if (ISPRINT)
			print(l, r);
	}

	void arrayAdd(T *l, T *r, T add)
	{
		for (int i(0); i < r-l; i++)
			l[i] += add;
		if (ISPRINT)
			print(l, r);
	}

	void assertSorted(T *l, T *r)
	{
		if (ISPRINT)
			print(l, r);
		for (int i(0); i < r-l-1; i++)
			if (l[i] > l[i+1]) {
				printf("Unsorted\n");
				exit(-1);
			}
	}

};

}

#endif