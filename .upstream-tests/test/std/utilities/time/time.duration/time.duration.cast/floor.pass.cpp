//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <cuda/std/chrono>

// floor

// template <class ToDuration, class Rep, class Period>
//   constexpr
//   ToDuration
//   floor(const duration<Rep, Period>& d);

#include <cuda/std/chrono>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

template <class ToDuration, class FromDuration>
__host__ __device__
void
test(const FromDuration& f, const ToDuration& d)
{
    {
    typedef decltype(cuda::std::chrono::floor<ToDuration>(f)) R;
    static_assert((cuda::std::is_same<R, ToDuration>::value), "");
    assert(cuda::std::chrono::floor<ToDuration>(f) == d);
    }
}

int main(int, char**)
{
//  7290000ms is 2 hours, 1 minute, and 30 seconds
    test(cuda::std::chrono::milliseconds( 7290000), cuda::std::chrono::hours( 2));
    test(cuda::std::chrono::milliseconds(-7290000), cuda::std::chrono::hours(-3));
    test(cuda::std::chrono::milliseconds( 7290000), cuda::std::chrono::minutes( 121));
    test(cuda::std::chrono::milliseconds(-7290000), cuda::std::chrono::minutes(-122));

    {
//  9000000ms is 2 hours and 30 minutes
    constexpr cuda::std::chrono::hours h1 = cuda::std::chrono::floor<cuda::std::chrono::hours>(cuda::std::chrono::milliseconds(9000000));
    static_assert(h1.count() == 2, "");
    constexpr cuda::std::chrono::hours h2 = cuda::std::chrono::floor<cuda::std::chrono::hours>(cuda::std::chrono::milliseconds(-9000000));
    static_assert(h2.count() == -3, "");
    }

  return 0;
}
