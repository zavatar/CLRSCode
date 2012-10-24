//////////////////////////////////////////////////////////////////////////
//
//	ModernC++.h
//
//	Written by Meng Zhu ,Zhejiang University (zhumeng1989@gmail.com)
//	Creation date: Oct.19 2012, 16:46
//
//	All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////

#ifndef ModernCpp_H
#define ModernCpp_H

#include <cassert>
#include <boost/static_assert.hpp>
#include <loki/static_check.h>
#include <loki/Typelist.h>
#include <loki/TypelistMacros.h>
#include <loki/HierarchyGenerators.h>
#include <loki/SmallObj.h>
#include <loki/Functor.h>

namespace mc {


// Chapter 1. Policy-Based Class Design

// Chapter 2. Techniques

	// 2.1 Compile-Time Assertions
	BOOST_STATIC_ASSERT(true);
#ifdef MCXX11
	// C++11
	static_assert(true, "Msg");
#endif
	// Loki Error?
	//LOKI_STATIC_CHECK(true, "Msg");
	
	// 2.3 Local Classes
	class Interface 
	{ 
	public: 
		virtual void Fun() = 0; 
	};
	// Adapter
	template <class T, class P> 
	Interface* MakeAdapter(const T& obj, const P& arg) 
	{ 
		class Local : public Interface 
		{ 
		public: 
			Local(const T& obj, const P& arg) 
				: obj_(obj), arg_(arg) {} 
			virtual void Fun() 
			{ 
				obj_.Call(arg_); 
			} 
		private: 
			T obj_; 
			P arg_; 
		}; 
		return new Local(obj, arg); 
	}

	// 2.6 Type Selection
	template <bool flag, typename T, typename U> 
	struct Select 
	{
		typedef T Result; 
	}; 
	template <typename T, typename U> 
	struct Select<false, T, U> 
	{ 
		typedef U Result; 
	};
	template <typename T, bool isPolymorphic> 
	class NiftyContainer 
	{ 
		typedef typename Select<isPolymorphic, T*, T>::Result 
			ValueType; 
	}; 

	// 2.7 Detecting Convertibility and Inheritance at Compile Time
	template <class T, class U> 
	class Conversion 
	{ 
		typedef char Small; 
		class Big { char dummy[2]; }; 
		static Small Test(U);
		static Big Test(...); 
		static T MakeT(); 
	public: 
		enum { exists = 
			sizeof(Test(MakeT())) == sizeof(Small) }; 
		enum { sameType = false };
	};
	template <class T> 
	class Conversion<T, T> 
	{ 
	public: 
		enum { exists = 1, sameType = 1 }; 
	}; 
#define SUPERSUBCLASS(T, U) \
	(::mc::Conversion<const U*, const T*>::exists && \
		!::mc::Conversion<const T*, const void*>::sameType)
#define SUPERSUBCLASS_STRICT(T, U) \
		(SUPERSUBCLASS(T, U) && \
		!::mc::Conversion<const T, const U>::sameType)

	

}

#endif