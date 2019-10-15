//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

#include <cuda/std/cassert>
#include <cuda/std/unordered_map>

// <cuda/std/unordered_map>

// bool contains(const key_type& x) const;

template <typename T, typename P, typename B, typename... Pairs>
void test(B bad, Pairs... args) {
    T map;
    P pairs[] = {args...};

    for (auto& p : pairs) map.insert(p);
    for (auto& p : pairs) assert(map.contains(p.first));

    assert(!map.contains(bad));
}

struct E { int a = 1; double b = 1; char c = 1; };

int main(int, char**)
{
    {
        test<cuda::std::unordered_map<char, int>, cuda::std::pair<char, int> >(
            'e', cuda::std::make_pair('a', 10), cuda::std::make_pair('b', 11),
            cuda::std::make_pair('c', 12), cuda::std::make_pair('d', 13));

        test<cuda::std::unordered_map<char, char>, cuda::std::pair<char, char> >(
            'e', cuda::std::make_pair('a', 'a'), cuda::std::make_pair('b', 'a'),
            cuda::std::make_pair('c', 'a'), cuda::std::make_pair('d', 'b'));

        test<cuda::std::unordered_map<int, E>, cuda::std::pair<int, E> >(
            -1, cuda::std::make_pair(1, E{}), cuda::std::make_pair(2, E{}),
            cuda::std::make_pair(3, E{}), cuda::std::make_pair(4, E{}));
    }
    {
        test<cuda::std::unordered_multimap<char, int>, cuda::std::pair<char, int> >(
            'e', cuda::std::make_pair('a', 10), cuda::std::make_pair('b', 11),
            cuda::std::make_pair('c', 12), cuda::std::make_pair('d', 13));

        test<cuda::std::unordered_multimap<char, char>, cuda::std::pair<char, char> >(
            'e', cuda::std::make_pair('a', 'a'), cuda::std::make_pair('b', 'a'),
            cuda::std::make_pair('c', 'a'), cuda::std::make_pair('d', 'b'));

        test<cuda::std::unordered_multimap<int, E>, cuda::std::pair<int, E> >(
            -1, cuda::std::make_pair(1, E{}), cuda::std::make_pair(2, E{}),
            cuda::std::make_pair(3, E{}), cuda::std::make_pair(4, E{}));
    }

    return 0;
}

