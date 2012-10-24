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
#include <algorithm>

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

template <typename T>
T maxsum3(T *x, int l, int u, int &left, int &right)
{
	left = l; right = u;
	if (l > u)
		return 0;
	if (l == u)
		return std::max(T(0), x[l]);

	int m = (l+u)/2;
	T lmax = 0;
	T sum = 0;
	for (int i(m); i >= l; i--)
	{
		sum += x[i];
		if (sum > lmax)
		{
			lmax = sum;
			left = i;
		}
	}
	T rmax = 0;
	sum = 0;
	for (int i(m+1); i <= u; i++)
	{
		sum += x[i];
		if (sum > rmax)
		{
			rmax = sum;
			right = i;
		}
	}
	T csum = lmax + rmax;
	int ll, lr;
	T lsum = maxsum3(x, l, m, ll, lr);
	int rl, rr;
	T rsum = maxsum3(x, m+1, u, rl, rr);
	if (lsum >= rsum && lsum >= csum)
	{
		left = ll;
		right = lr;
		return lsum;
	} else if (rsum >= lsum && rsum >= csum)
	{
		left = rl;
		right = rr;
		return rsum;
	} else
		return csum;
}

template <typename T>
T maxsum4(T *x, int n, int &left, int &right)
{
	T maxsofar = 0;
	T maxendinghere = 0;
	int t = left = right = 0;
	for (int i(0); i < n; i++)
	{
		maxendinghere += x[i];
		if (maxendinghere < 0)
		{
			t = i+1;
			maxendinghere = 0;
		}
		if (maxendinghere > maxsofar)
		{
			left = t;
			right = i;
			maxsofar = maxendinghere;
		}
	}
	return maxsofar;
}

} // namespace

#endif