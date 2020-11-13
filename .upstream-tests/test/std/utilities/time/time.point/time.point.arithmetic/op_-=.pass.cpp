//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// time_point

// time_point& operator-=(const duration& d);
// constexpr in c++17

#include <cuda/std/chrono>
#include <cuda/std/cassert>

#include "test_macros.h"

#if TEST_STD_VER > 14
constexpr bool constexpr_test()
{
    typedef cuda::std::chrono::system_clock Clock;
    typedef cuda::std::chrono::milliseconds Duration;
    cuda::std::chrono::time_point<Clock, Duration> t(Duration(5));
    t -= Duration(4);
    return t.time_since_epoch() == Duration(1);
}
#endif

int main(int, char**)
{
    {
    typedef cuda::std::chrono::system_clock Clock;
    typedef cuda::std::chrono::milliseconds Duration;
    cuda::std::chrono::time_point<Clock, Duration> t(Duration(3));
    t -= Duration(2);
    assert(t.time_since_epoch() == Duration(1));
    }

#if TEST_STD_VER > 14
    static_assert(constexpr_test(), "");
#endif

  return 0;
}
