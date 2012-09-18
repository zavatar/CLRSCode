//////////////////////////////////////////////////////////////////////////
//
//	PgrogrammingPearls.h
//
//	Written by Meng Zhu ,Zhejiang University (zhumeng1989@gmail.com)
//	Creation date: Sep.18 2012, 12:54
//
//	All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////

#ifndef PgrogrammingPearls_H
#define PgrogrammingPearls_H

#include "Introduction2Algorithms.h"

namespace pp {

// NOTE: 
// 9.5 Problems 4
template <typename T>
T arrmax(T *x, int n)
{
	if (n == 1)
		return x[0];
	else
		return std::max(x[n-1], arrmax(x, n-1));
}

template <typename T>
T arrmaxMacro(T *x, int n)
{
	if (n == 1)
		return x[0];
	else
#define ppmax(a, b) ((a) > (b) ? (a) : (b))
		return ppmax(x[n-1], arrmaxMacro(x, n-1));
#undef ppmax
}

template <typename T>
void isort1(T *x, int n)
{
	for (int i(1); i < n; i++)
		for (int j(i); j > 0 && x[j-1] > x[j]; j--)
			std::swap(x[j-1], x[j]);
}

template <typename T>
void isort2(T *x, int n)
{
	for (int i(1); i < n; i++)
		for (int j(i); j > 0 && x[j-1] > x[j]; j--)
		{T tmp = x[j-1]; x[j-1] = x[j]; x[j] = tmp;}
}

template <typename T>
void isort3(T *x, int n)
{
	for (int i(1); i < n; i++)
	{
		T t = x[i];
		int j = i;
		for (; j > 0 && x[j-1] > t; j--)
			x[j] = x[j-1];
		x[j] = t;
	}
}

} // namespace

#endif