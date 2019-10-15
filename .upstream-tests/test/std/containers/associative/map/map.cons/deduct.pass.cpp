//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/map>
// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: libcpp-no-deduction-guides

// template<class InputIterator,
//          class Compare = less<iter-value-type<InputIterator>>,
//          class Allocator = allocator<iter-value-type<InputIterator>>>
// map(InputIterator, InputIterator,
//          Compare = Compare(), Allocator = Allocator())
//   -> map<iter-value-type<InputIterator>, Compare, Allocator>;
// template<class Key, class Compare = less<Key>, class Allocator = allocator<Key>>
// map(initializer_list<Key>, Compare = Compare(), Allocator = Allocator())
//   -> map<Key, Compare, Allocator>;
// template<class InputIterator, class Allocator>
// map(InputIterator, InputIterator, Allocator)
//   -> map<iter-value-type<InputIterator>, less<iter-value-type<InputIterator>>, Allocator>;
// template<class Key, class Allocator>
// map(initializer_list<Key>, Allocator)
//   -> map<Key, less<Key>, Allocator>;

#include <cuda/std/algorithm> // cuda::std::equal
#include <cuda/std/cassert>
#include <cuda/std/climits> // INT_MAX
#include <cuda/std/functional>
#include <cuda/std/map>
#include <cuda/std/type_traits>

#include "test_allocator.h"

using P = cuda::std::pair<int, long>;
using PC = cuda::std::pair<const int, long>;

int main(int, char**)
{
    {
    const P arr[] = { {1,1L}, {2,2L}, {1,1L}, {INT_MAX,1L}, {3,1L} };
    cuda::std::map m(cuda::std::begin(arr), cuda::std::end(arr));

    ASSERT_SAME_TYPE(decltype(m), cuda::std::map<int, long>);
    const PC expected_m[] = { {1,1L}, {2,2L}, {3,1L}, {INT_MAX,1L} };
    assert(cuda::std::equal(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    }

    {
    const P arr[] = { {1,1L}, {2,2L}, {1,1L}, {INT_MAX,1L}, {3,1L} };
    cuda::std::map m(cuda::std::begin(arr), cuda::std::end(arr), cuda::std::greater<int>());

    ASSERT_SAME_TYPE(decltype(m), cuda::std::map<int, long, cuda::std::greater<int>>);
    const PC expected_m[] = { {INT_MAX,1L}, {3,1L}, {2,2L}, {1,1L} };
    assert(cuda::std::equal(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    }

    {
    const P arr[] = { {1,1L}, {2,2L}, {1,1L}, {INT_MAX,1L}, {3,1L} };
    cuda::std::map m(cuda::std::begin(arr), cuda::std::end(arr), cuda::std::greater<int>(), test_allocator<PC>(0, 42));

    ASSERT_SAME_TYPE(decltype(m), cuda::std::map<int, long, cuda::std::greater<int>, test_allocator<PC>>);
    const PC expected_m[] = { {INT_MAX,1L}, {3,1L}, {2,2L}, {1,1L} };
    assert(cuda::std::equal(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    assert(m.get_allocator().get_id() == 42);
    }

    {
    cuda::std::map<int, long> source;
    cuda::std::map m(source);
    ASSERT_SAME_TYPE(decltype(m), decltype(source));
    assert(m.size() == 0);
    }

    {
    cuda::std::map<int, long> source;
    cuda::std::map m{source};  // braces instead of parens
    ASSERT_SAME_TYPE(decltype(m), decltype(source));
    assert(m.size() == 0);
    }

    {
    cuda::std::map<int, long> source;
    cuda::std::map m(source, cuda::std::map<int, long>::allocator_type());
    ASSERT_SAME_TYPE(decltype(m), decltype(source));
    assert(m.size() == 0);
    }

    {
    cuda::std::map m{ P{1,1L}, P{2,2L}, P{1,1L}, P{INT_MAX,1L}, P{3,1L} };

    ASSERT_SAME_TYPE(decltype(m), cuda::std::map<int, long>);
    const PC expected_m[] = { {1,1L}, {2,2L}, {3,1L}, {INT_MAX,1L} };
    assert(cuda::std::equal(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    }

    {
    cuda::std::map m({ P{1,1L}, P{2,2L}, P{1,1L}, P{INT_MAX,1L}, P{3,1L} }, cuda::std::greater<int>());

    ASSERT_SAME_TYPE(decltype(m), cuda::std::map<int, long, cuda::std::greater<int>>);
    const PC expected_m[] = { {INT_MAX,1L}, {3,1L}, {2,2L}, {1,1L} };
    assert(cuda::std::equal(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    }

    {
    cuda::std::map m({ P{1,1L}, P{2,2L}, P{1,1L}, P{INT_MAX,1L}, P{3,1L} }, cuda::std::greater<int>(), test_allocator<PC>(0, 43));

    ASSERT_SAME_TYPE(decltype(m), cuda::std::map<int, long, cuda::std::greater<int>, test_allocator<PC>>);
    const PC expected_m[] = { {INT_MAX,1L}, {3,1L}, {2,2L}, {1,1L} };
    assert(cuda::std::equal(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    assert(m.get_allocator().get_id() == 43);
    }

    {
    const P arr[] = { {1,1L}, {2,2L}, {1,1L}, {INT_MAX,1L}, {3,1L} };
    cuda::std::map m(cuda::std::begin(arr), cuda::std::end(arr), test_allocator<PC>(0, 44));

    ASSERT_SAME_TYPE(decltype(m), cuda::std::map<int, long, cuda::std::less<int>, test_allocator<PC>>);
    const PC expected_m[] = { {1,1L}, {2,2L}, {3,1L}, {INT_MAX,1L} };
    assert(cuda::std::equal(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    assert(m.get_allocator().get_id() == 44);
    }

    {
    cuda::std::map m({ P{1,1L}, P{2,2L}, P{1,1L}, P{INT_MAX,1L}, P{3,1L} }, test_allocator<PC>(0, 45));

    ASSERT_SAME_TYPE(decltype(m), cuda::std::map<int, long, cuda::std::less<int>, test_allocator<PC>>);
    const PC expected_m[] = { {1,1L}, {2,2L}, {3,1L}, {INT_MAX,1L} };
    assert(cuda::std::equal(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    assert(m.get_allocator().get_id() == 45);
    }

    return 0;
}
