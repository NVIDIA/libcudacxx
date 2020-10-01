//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<> class complex<float>
// {
// public:
//     explicit constexpr complex(const complex<long double>&);
// };

#include <cuda/std/complex>
#include <cuda/std/cassert>

int main(int, char**)
{
    const cuda::std::complex<long double> cd(2.5, 3.5);
    cuda::std::complex<float> cf = cd;
    assert(cf.real() == cd.real());
    assert(cf.imag() == cd.imag());

  return 0;
}
