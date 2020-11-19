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
//   conj(const complex<T>& x);

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"

template <class T>
void
test(const cuda::std::complex<T>& z, cuda::std::complex<T> x)
{
    assert(conj(z) == x);
}

template <class T>
void
test()
{
    test(cuda::std::complex<T>(1, 2), cuda::std::complex<T>(1, -2));
    test(cuda::std::complex<T>(-1, 2), cuda::std::complex<T>(-1, -2));
    test(cuda::std::complex<T>(1, -2), cuda::std::complex<T>(1, 2));
    test(cuda::std::complex<T>(-1, -2), cuda::std::complex<T>(-1, 2));
}

int main(int, char**)
{
    test<float>();
    test<double>();
    test<long double>();

  return 0;
}
