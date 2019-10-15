//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

#include <cuda/std/cassert>
#include <cuda/std/set>

// <cuda/std/set>

// bool contains(const key_type& x) const;

template <typename T, typename V, typename B, typename... Vals>
void test(B bad, Vals... args) {
    T set;
    V vals[] = {args...};

    for (auto& v : vals) set.insert(v);
    for (auto& v : vals) assert(set.contains(v));

    assert(!set.contains(bad));
}

struct E { int a = 1; double b = 1; char c = 1; };

int main(int, char**)
{
    {
        test<cuda::std::set<int>, int>(14, 10, 11, 12, 13);
        test<cuda::std::set<char>, char>('e', 'a', 'b', 'c', 'd');
    }
    {
        test<cuda::std::multiset<int>, int>(14, 10, 11, 12, 13);
        test<cuda::std::multiset<char>, char>('e', 'a', 'b', 'c', 'd');
    }

    return 0;
}

