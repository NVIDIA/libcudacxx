//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>

// using weeks = duration<signed integer type of at least 22 bits, ratio_multiply<ratio<7>, days::period>>;

#include <cuda/std/chrono>
#include <cuda/std/type_traits>
#include <cuda/std/limits>

#include "test_macros.h"

int main(int, char**)
{
    typedef cuda::std::chrono::weeks D;
    typedef D::rep Rep;
    typedef D::period Period;
    static_assert(cuda::std::is_signed<Rep>::value, "");
    static_assert(cuda::std::is_integral<Rep>::value, "");
    static_assert(cuda::std::numeric_limits<Rep>::digits >= 22, "");
    static_assert(cuda::std::is_same_v<Period, cuda::std::ratio_multiply<cuda::std::ratio<7>, cuda::std::chrono::days::period>>, "");

  return 0;
}
