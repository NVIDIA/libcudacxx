//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <cuda/std/chrono>

#include <cuda/std/complex>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    using namespace cuda::std::literals::complex_literals;

//  Make sure the types are right
    static_assert ( cuda::std::is_same<decltype( 3.0il ), cuda::std::complex<long double>>::value, "" );
    static_assert ( cuda::std::is_same<decltype( 3il   ), cuda::std::complex<long double>>::value, "" );
    static_assert ( cuda::std::is_same<decltype( 3.0i  ), cuda::std::complex<double>>::value, "" );
    static_assert ( cuda::std::is_same<decltype( 3i    ), cuda::std::complex<double>>::value, "" );
    static_assert ( cuda::std::is_same<decltype( 3.0if ), cuda::std::complex<float>>::value, "" );
    static_assert ( cuda::std::is_same<decltype( 3if   ), cuda::std::complex<float>>::value, "" );

    {
    cuda::std::complex<long double> c1 = 3.0il;
    assert ( c1 == cuda::std::complex<long double>(0, 3.0));
    auto c2 = 3il;
    assert ( c1 == c2 );
    }

    {
    cuda::std::complex<double> c1 = 3.0i;
    assert ( c1 == cuda::std::complex<double>(0, 3.0));
    auto c2 = 3i;
    assert ( c1 == c2 );
    }

    {
    cuda::std::complex<float> c1 = 3.0if;
    assert ( c1 == cuda::std::complex<float>(0, 3.0));
    auto c2 = 3if;
    assert ( c1 == c2 );
    }

  return 0;
}
