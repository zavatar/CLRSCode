For Chinese, please check my Blog: 
http://blog.csdn.net/zhumeng1989/article/category/1232333

1. 项目中的函数实现都尽可能同原文一致，方便对照。包括名字大写这点。但由于原
文数组索引从1开始，这点注意区别。
2. 加入 Programming Pearls 编程珠玑 部分内容。
3. 加入 Modern C++ Design C++设计新思维 部分内容, 下载loki-0.1.7 http://sourceforge.net/projects/loki-lib/ 并修改include路径，下载boost_1_51_0 http://www.boost.org/。Note: With new C++11, The template Typelist could be implemented with 'variadic template'[http://stackoverflow.com/questions/9463659/wrap-lokitypelist-with-c11-variadic-template] AND Type traits have been added in C++11.
4. Added Parallel Forall Part, which is come from Nvidia Developer Blog, https://developer.nvidia.com/blog. This part is aimed to follow their "Programming CUDA with C and C++" series, using CUDA 5. And, due to the power of GPU parallel programming, I plain adding  some comparations to basic Algorithm, such as Sort, Sum and so on.
5. Added Thrust CUDA library.
6. Added BSGP, about BSGP http://www.houqiming.net/

TODO: It will be better if separate template declaration and implement, even C++ DO NOT support only template declaration in header file, while we should use .inl file adding to the tail of .h for more clear hierarchy.

NOTE: I think the hierarchy of the code is bull shit, after I read something about the Design Patterns. Now, I'm still learning some basic knowledge. If time is ready, I will refactor these code.