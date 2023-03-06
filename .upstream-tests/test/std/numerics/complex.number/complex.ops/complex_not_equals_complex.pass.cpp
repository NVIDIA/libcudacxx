//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<class T>
//   bool
//   operator!=(const complex<T>& lhs, const complex<T>& rhs);

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wliteral-conversion"
#endif

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void
test_constexpr()
{
#if TEST_STD_VER > 11
    {
    constexpr cuda::std::complex<T> lhs(1.5,  2.5);
    constexpr cuda::std::complex<T> rhs(1.5, -2.5);
    static_assert(lhs != rhs, "");
    }
    {
    constexpr cuda::std::complex<T> lhs(1.5, 2.5);
    constexpr cuda::std::complex<T> rhs(1.5, 2.5);
    static_assert(!(lhs != rhs), "" );
    }
#endif
}


template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool
test()
{
    {
        cuda::std::complex<T> lhs(1.5,  2.5);
        cuda::std::complex<T> rhs(1.5, -2.5);
        assert(lhs != rhs);
    }
    {
        cuda::std::complex<T> lhs(1.5, 2.5);
        cuda::std::complex<T> rhs(1.5, 2.5);
        assert(!(lhs != rhs));
    }

    test_constexpr<T> ();

    return true;
}

int main(int, char**)
{
    test<float>();
    test<double>();
// CUDA treats long double as double
//  test<long double>();
#if TEST_STD_VER > 11 
    static_assert(test<float>(), "");
    static_assert(test<double>(), "");
// CUDA treats long double as double
//  static_assert(test<long double>(), "");
#endif

    test_constexpr<int> ();

    return 0;
}
