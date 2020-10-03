//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc

// <utility>

// template <class T1, class T2> struct pair

// template <class U, class V> pair(const pair<U, V>& p);

#include <cuda/std/utility>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef cuda::std::pair<int, short> P1;
        typedef cuda::std::pair<double, long> P2;
        const P1 p1(3, static_cast<short>(4));
        const P2 p2 = p1;
        assert(p2.first == 3);
        assert(p2.second == 4);
    }

  return 0;
}
