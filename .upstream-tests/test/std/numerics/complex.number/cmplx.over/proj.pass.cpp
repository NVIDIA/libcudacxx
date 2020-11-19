//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<class T>    complex<T>           proj(const complex<T>&);
//                      complex<long double> proj(long double);
//                      complex<double>      proj(double);
// template<Integral T> complex<double>      proj(T);
//                      complex<float>       proj(float);

#include <cuda/std/complex>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
void
test(T x, typename cuda::std::enable_if<cuda::std::is_integral<T>::value>::type* = 0)
{
    static_assert((cuda::std::is_same<decltype(cuda::std::proj(x)), cuda::std::complex<double> >::value), "");
    assert(cuda::std::proj(x) == proj(cuda::std::complex<double>(x, 0)));
}

template <class T>
void
test(T x, typename cuda::std::enable_if<cuda::std::is_floating_point<T>::value>::type* = 0)
{
    static_assert((cuda::std::is_same<decltype(cuda::std::proj(x)), cuda::std::complex<T> >::value), "");
    assert(cuda::std::proj(x) == proj(cuda::std::complex<T>(x, 0)));
}

template <class T>
void
test(T x, typename cuda::std::enable_if<!cuda::std::is_integral<T>::value &&
                                  !cuda::std::is_floating_point<T>::value>::type* = 0)
{
    static_assert((cuda::std::is_same<decltype(cuda::std::proj(x)), cuda::std::complex<T> >::value), "");
    assert(cuda::std::proj(x) == proj(cuda::std::complex<T>(x, 0)));
}

template <class T>
void
test()
{
    test<T>(0);
    test<T>(1);
    test<T>(10);
}

int main(int, char**)
{
    test<float>();
    test<double>();
    test<long double>();
    test<int>();
    test<unsigned>();
    test<long long>();

  return 0;
}
