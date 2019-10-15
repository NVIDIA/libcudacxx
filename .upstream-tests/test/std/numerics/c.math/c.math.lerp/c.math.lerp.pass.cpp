//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
// <cuda/std/cmath>

// constexpr float lerp(float a, float b, float t);
// constexpr double lerp(double a, double b, double t);
// constexpr long double lerp(long double a, long double b, long double t);


#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "fp_compare.h"

template <typename T>
constexpr bool constexpr_test()
{
    return cuda::std::lerp(T( 0), T(12), T(0))   == T(0)
        && cuda::std::lerp(T(12), T( 0), T(0.5)) == T(6)
        && cuda::std::lerp(T( 0), T(12), T(2))   == T(24);
}


template <typename T>
void test()
{
    ASSERT_SAME_TYPE(T, decltype(cuda::std::lerp(T(), T(), T())));
    LIBCPP_ASSERT_NOEXCEPT(      cuda::std::lerp(T(), T(), T()));

//     constexpr T minV = cuda::std::numeric_limits<T>::min();
    constexpr T maxV = cuda::std::numeric_limits<T>::max();
    constexpr T inf  = cuda::std::numeric_limits<T>::infinity();

//  Things that can be compared exactly
    assert((cuda::std::lerp(T( 0), T(12), T(0)) == T(0)));
    assert((cuda::std::lerp(T( 0), T(12), T(1)) == T(12)));
    assert((cuda::std::lerp(T(12), T( 0), T(0)) == T(12)));
    assert((cuda::std::lerp(T(12), T( 0), T(1)) == T(0)));
    
    assert((cuda::std::lerp(T( 0), T(12), T(0.5)) == T(6)));
    assert((cuda::std::lerp(T(12), T( 0), T(0.5)) == T(6)));
    assert((cuda::std::lerp(T( 0), T(12), T(2))   == T(24)));
    assert((cuda::std::lerp(T(12), T( 0), T(2))   == T(-12)));

    assert((cuda::std::lerp(maxV, maxV/10, T(0)) == maxV));
    assert((cuda::std::lerp(maxV/10, maxV, T(1)) == maxV));

    assert((cuda::std::lerp(T(2.3), T(2.3), inf) == T(2.3)));

    assert(cuda::std::lerp(T( 0), T( 0), T(23)) ==  T(0));
    assert(cuda::std::isnan(cuda::std::lerp(T( 0), T( 0), inf)));
}


int main(int, char**)
{
    static_assert(constexpr_test<float>(), "");
    static_assert(constexpr_test<double>(), "");
    static_assert(constexpr_test<long double>(), "");
    
    test<float>();
    test<double>();
    test<long double>();
    
    return 0;
}
