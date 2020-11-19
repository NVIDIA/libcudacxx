//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<class T>
//   complex<T>
//   operator/(const T& lhs, const complex<T>& rhs);

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"

template <class T>
__host__ __device__ void
test(const T& lhs, const cuda::std::complex<T>& rhs, cuda::std::complex<T> x)
{
    assert(lhs / rhs == x);
}

template <class T>
__host__ __device__ void
test()
{
    T lhs(-8.5);
    cuda::std::complex<T> rhs(1.5, 2.5);
    cuda::std::complex<T>   x(-1.5, 2.5);
    test(lhs, rhs, x);
}

int main(int, char**)
{
    test<float>();
    test<double>();
// CUDA treats long double as double
//  test<long double>();

  return 0;
}
