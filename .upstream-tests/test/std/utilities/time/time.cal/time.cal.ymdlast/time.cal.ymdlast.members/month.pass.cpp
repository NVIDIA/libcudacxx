//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class year_month_day_last;

// constexpr chrono::month month() const noexcept;
//  Returns: wd_

#include <cuda/std/chrono>
#include <cuda/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using year                = cuda::std::chrono::year;
    using month               = cuda::std::chrono::month;
    using month_day_last      = cuda::std::chrono::month_day_last;
    using year_month_day_last = cuda::std::chrono::year_month_day_last;

    ASSERT_NOEXCEPT(                 std::declval<const year_month_day_last>().month());
    ASSERT_SAME_TYPE(month, decltype(cuda::std::declval<const year_month_day_last>().month()));

    for (unsigned i = 1; i <= 50; ++i)
    {
        year_month_day_last ymd(year{1234}, month_day_last{month{i}});
        assert( static_cast<unsigned>(ymd.month()) == i);
    }

  return 0;
}
