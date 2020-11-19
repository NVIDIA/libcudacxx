//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<> class complex<double>
// {
// public:
//     constexpr complex(const complex<float>&);
// };

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
    const cuda::std::complex<float> cd(2.5, 3.5);
    cuda::std::complex<double> cf = cd;
    assert(cf.real() == cd.real());
    assert(cf.imag() == cd.imag());
    }
#if TEST_STD_VER >= 11
    {
    constexpr cuda::std::complex<float> cd(2.5, 3.5);
    constexpr cuda::std::complex<double> cf = cd;
    static_assert(cf.real() == cd.real(), "");
    static_assert(cf.imag() == cd.imag(), "");
    }
#endif

  return 0;
}
