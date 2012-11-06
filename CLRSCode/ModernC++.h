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
#include <boost/function.hpp>
#include <loki/static_check.h>
#include <loki/Typelist.h>
#include <loki/TypelistMacros.h>
#include <loki/HierarchyGenerators.h>
#include <loki/SmallObj.h>
#include <loki/Functor.h>
#include <loki/EmptyType.h>
#include <loki/TypeTraits.h>

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

// Using
namespace mc {

	namespace TlPrivate {
#ifdef MCXX11
		// C++11, need compiling with g++ -std=c++0x
		template<typename T>  
		void output(T&& value)  
		{  
			std::cout << value << std::endl;
		}
		template<typename First, typename... Rest>  
		void output(First&& first, Rest&&... rest)  
		{  
			std::cout << first << ",";
			output(std::forward<Rest>(rest)...);
		}
		template<typename T>  
		T sum(T&& value)  
		{  
			return value;  
		}  
		template<typename First, typename Second, typename... Rest>  
		First sum(First&& first, Second&& second, Rest&&... rest)  
		{  
			return sum(first + second, std::forward<Rest>(rest)...);  
		}
#endif
		template <class T> 
		struct Holder 
		{
			//#pragma message("Holder")
			T value_;
		}; 

		class BBase {};
		class Window {};
		class Button {};
		class ScrollBar {};

		template <class T, class Base> 
		class EventHandler : public Base 
		{ 
		public: 
			virtual void OnEvent(T& obj, int eventId); 
		};

	}

	void Typelist_Benchmark()
	{
#ifdef MCXX11
		// C++11, need compiling with g++ -std=c++0x
		double d = 2.3;  
		TlPrivate::output(42,"hello",d,'a');  
		TlPrivate::output("hello",d,'a');  
		TlPrivate::output(d,'a');  
		TlPrivate::output('a');  
		TlPrivate::output(TlPrivate::sum(1), TlPrivate::sum(1, 2), TlPrivate::sum(1, 2, 3));
#endif
		typedef Loki::TL::MakeTypelist<bool, char, int, float, double>::Result MyTypelist;

		typedef Loki::TL::Reverse<MyTypelist>::Result MyTypelistReversed;

		typedef Loki::TL::Append<MyTypelist, MyTypelistReversed>::Result MyTypelistSymmetrical;

		std::cout << Loki::TL::Length<MyTypelistReversed>::value << std::endl;

		std::cout << Loki::TL::IndexOf<MyTypelistSymmetrical, float>::value << std::endl <<
			typeid(Loki::TL::TypeAt<MyTypelistSymmetrical, 6>::Result).name() << std::endl;

		// 3.13.1 Generating Scattered Hierarchies
		typedef Loki::GenScatterHierarchy
			<
			Loki::TL::MakeTypelist
			<
			bool,
			char,
			int
			>::Result,
			TlPrivate::Holder
			>
			MyScatterHierarchy;
		// 3*4
		std::cout << sizeof(MyScatterHierarchy) << std::endl;

		// Deal with ambiguity using NoDuplicates
		Loki::GenScatterHierarchy
			<
			Loki::TL::NoDuplicates
			<
			Loki::TL::MakeTypelist
			<
			int,
			int,
			int
			>::Result
			>::Result,
			TlPrivate::Holder
			> ma0;
		Loki::Field<int>(ma0);

		Loki::Tuple<Loki::TL::NoDuplicates<MyTypelistSymmetrical>::Result> ma1;
		Loki::Field<int>(ma1);

		// 3.13.2 Generating Tuples
		typedef Loki::Tuple<MyTypelist> MyTuple;
		MyTuple mt;
		Loki::Field<0>(mt) = true;
		Loki::Field<1>(mt) = 'A';
		Loki::Field<2>(mt) = 256;
		Loki::Field<3>(mt) = 1.f;


		// 3.13.3 Generating Linear Hierarchies
		typedef Loki::GenLinearHierarchy
			< 
			Loki::TL::MakeTypelist
			<
			TlPrivate::Window,
			TlPrivate::Button,
			TlPrivate::ScrollBar
			>::Result,
			TlPrivate::EventHandler,
			TlPrivate::BBase
			> 
			MyEventHandler;
		// 1*4
		std::cout << sizeof(MyEventHandler) << std::endl;
	}

	namespace FunPrivate {

		void TestFunction(int);
		void TestFunction(int i, double d)
		{
			std::cout << "TestFunction(" << i 
				<< ", " << d << ") called.\n";
		}

		struct TestFunctor
		{
			void operator()(int i, double d) 
			{ 
				std::cout << "TestFunctor::operator()(" << i 
					<< ", " << d << ") called.\n"; 
			}
		};

		template <typename T>
		class Functor
		{
		public:
			// const is important
			Functor(const T& fun) : _f(fun) {}

			void operator()(int i, double d)
			{
				_f(i, d);
			}

		private:
			T _f;
		};

		class Parrot 
		{ 
		public: 
			void Eat() 
			{ 
				std::cout << "Tsk, knick, tsk...\n"; 
			} 
			void Speak() 
			{ 
				std::cout << "Oh Captain, my Captain!\n"; 
			} 
		};

	}

	void Functor_Benchmark()
	{
		// Loki::Functor
		// Function
		//Loki::Functor<void, LOKI_TYPELIST_2(int, double)> cmd1(FunPrivate::TestFunction);
		//cmd1(4, 4.5);
		// operate()
		FunPrivate::TestFunctor f;
		Loki::Functor<void, LOKI_TYPELIST_2(int, double)> cmd2(f);
		cmd2(4, 4.5);
		// Functor
		Loki::Functor<void, LOKI_TYPELIST_2(int, double)> cmd3(cmd2);
		cmd3(4, 4.5);
		// FunPrivate::Functor
		// 	// Function
		// 	FunPrivate::Functor<void(*)(int, double)> fun1(FunPrivate::TestFunction);
		// 	fun1(4, 4.5);
		// operate()
		FunPrivate::Functor<FunPrivate::TestFunctor> fun2(f);
		fun2(4, 4.5);
		// Functor
		FunPrivate::Functor<FunPrivate::Functor<FunPrivate::TestFunctor>> fun3(fun2);
		fun3(4, 4.5);

		typedef void (*TpFun)(int, double);
		// C++11 Alias templates
		//using TpFun = void(*)(int, double);
		// Method 1:
		TpFun pF = FunPrivate::TestFunction;
		Loki::Functor<void, LOKI_TYPELIST_2(int, double)> cmd4(pF);
		cmd4(4, 4.5);
		// Method 2:
		Loki::Functor<void, LOKI_TYPELIST_2(int, double)> cmd5(
			static_cast<TpFun>(FunPrivate::TestFunction));
		cmd5(4, 4.5);

		typedef void (FunPrivate::Parrot::* TpMemFun)();
		TpMemFun pActivity = &FunPrivate::Parrot::Eat;
		FunPrivate::Parrot geronimo; 
		FunPrivate::Parrot* pGeronimo = &geronimo;
		(geronimo.*pActivity)(); 
		(pGeronimo->*pActivity)();
		pActivity = &FunPrivate::Parrot::Speak;
		(geronimo.*pActivity)();

		Loki::Functor<> cmd6(&geronimo, &FunPrivate::Parrot::Eat),
			cmd7(&geronimo, &FunPrivate::Parrot::Speak);
		cmd6();
		cmd7();
	}

}


#endif