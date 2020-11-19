//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<Arithmetic T>
//   T
//   arg(T x);

#include <cuda/std/complex>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>


#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ void
test(T x, typename cuda::std::enable_if<cuda::std::is_integral<T>::value>::type* = 0)
{
    static_assert((cuda::std::is_same<decltype(cuda::std::arg(x)), double>::value), "");
    assert(cuda::std::arg(x) == arg(cuda::std::complex<double>(static_cast<double>(x), 0)));
}

template <class T>
__host__ __device__ void
test(T x, typename cuda::std::enable_if<!cuda::std::is_integral<T>::value>::type* = 0)
{
    static_assert((cuda::std::is_same<decltype(cuda::std::arg(x)), T>::value), "");
    assert(cuda::std::arg(x) == arg(cuda::std::complex<T>(x, 0)));
}

template <class T>
__host__ __device__ void test()
{
    test<T>(0);
    test<T>(1);
    test<T>(10);
}

int main(int, char**)
{
    test<float>();
    test<double>();
// CUDA treats long double as double
//  test<long double>();
    test<int>();
    test<unsigned>();
    test<long long>();

  return 0;
}
