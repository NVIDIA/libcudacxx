//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/unordered_map>
// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: libcpp-no-deduction-guides

// template<class InputIterator,
//          class Hash = hash<iter-key-type<InputIterator>>,
//          class Pred = equal_to<iter-key-type<InputIterator>>,
//          class Allocator = allocator<iter-to-alloc-type<InputIterator>>>
// unordered_map(InputIterator, InputIterator, typename see below::size_type = see below,
//               Hash = Hash(), Pred = Pred(), Allocator = Allocator())
//   -> unordered_map<iter-key-type<InputIterator>, iter-mapped-type<InputIterator>, Hash, Pred,
//                    Allocator>;
//
// template<class Key, class T, class Hash = hash<Key>,
//          class Pred = equal_to<Key>, class Allocator = allocator<pair<const Key, T>>>
// unordered_map(initializer_list<pair<Key, T>>,
//               typename see below::size_type = see below, Hash = Hash(),
//               Pred = Pred(), Allocator = Allocator())
//   -> unordered_map<Key, T, Hash, Pred, Allocator>;
//
// template<class InputIterator, class Allocator>
// unordered_map(InputIterator, InputIterator, typename see below::size_type, Allocator)
//   -> unordered_map<iter-key-type<InputIterator>, iter-mapped-type<InputIterator>,
//                    hash<iter-key-type<InputIterator>>,
//                    equal_to<iter-key-type<InputIterator>>, Allocator>;
//
// template<class InputIterator, class Allocator>
// unordered_map(InputIterator, InputIterator, Allocator)
//   -> unordered_map<iter-key-type<InputIterator>, iter-mapped-type<InputIterator>,
//                    hash<iter-key-type<InputIterator>>,
//                    equal_to<iter-key-type<InputIterator>>, Allocator>;
//
// template<class InputIterator, class Hash, class Allocator>
// unordered_map(InputIterator, InputIterator, typename see below::size_type, Hash, Allocator)
//   -> unordered_map<iter-key-type<InputIterator>, iter-mapped-type<InputIterator>, Hash,
//                    equal_to<iter-key-type<InputIterator>>, Allocator>;
//
// template<class Key, class T, class Allocator>
// unordered_map(initializer_list<pair<Key, T>>, typename see below::size_type, Allocator)
//   -> unordered_map<Key, T, hash<Key>, equal_to<Key>, Allocator>;
//
// template<class Key, class T, class Allocator>
// unordered_map(initializer_list<pair<Key, T>>, Allocator)
//   -> unordered_map<Key, T, hash<Key>, equal_to<Key>, Allocator>;
//
// template<class Key, class T, class Hash, class Allocator>
// unordered_map(initializer_list<pair<Key, T>>, typename see below::size_type, Hash,
//               Allocator)
//   -> unordered_map<Key, T, Hash, equal_to<Key>, Allocator>;

#include <cuda/std/algorithm> // is_permutation
#include <cuda/std/cassert>
#include <cuda/std/climits> // INT_MAX
#include <cuda/std/type_traits>
#include <cuda/std/unordered_map>

#include "test_allocator.h"

using P = cuda::std::pair<int, long>;
using PC = cuda::std::pair<const int, long>;

int main(int, char**)
{
    const PC expected_m[] = { {1,1}, {2,2}, {3,1}, {INT_MAX,1} };

    {
    const P arr[] = { {1,1}, {2,2}, {1,1}, {INT_MAX,1}, {3,1} };
    cuda::std::unordered_map m(cuda::std::begin(arr), cuda::std::end(arr));
    ASSERT_SAME_TYPE(decltype(m), cuda::std::unordered_map<int, long>);
    assert(cuda::std::is_permutation(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    }

    {
    const P arr[] = { {1,1}, {2,2}, {1,1}, {INT_MAX,1}, {3,1} };
    cuda::std::unordered_map m(cuda::std::begin(arr), cuda::std::end(arr), 42);
    ASSERT_SAME_TYPE(decltype(m), cuda::std::unordered_map<int, long>);
    assert(cuda::std::is_permutation(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    }

    {
    const P arr[] = { {1,1}, {2,2}, {1,1}, {INT_MAX,1}, {3,1} };
    cuda::std::unordered_map m(cuda::std::begin(arr), cuda::std::end(arr), 42, cuda::std::hash<short>());
    ASSERT_SAME_TYPE(decltype(m), cuda::std::unordered_map<int, long, cuda::std::hash<short>, cuda::std::equal_to<int>>);
    assert(cuda::std::is_permutation(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    }

    {
    const P arr[] = { {1,1}, {2,2}, {1,1}, {INT_MAX,1}, {3,1} };
    cuda::std::unordered_map m(cuda::std::begin(arr), cuda::std::end(arr), 42, cuda::std::hash<short>(), cuda::std::equal_to<>());
    ASSERT_SAME_TYPE(decltype(m), cuda::std::unordered_map<int, long, cuda::std::hash<short>, cuda::std::equal_to<>>);
    assert(cuda::std::is_permutation(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    }

    {
    const P arr[] = { {1,1}, {2,2}, {1,1}, {INT_MAX,1}, {3,1} };
    cuda::std::unordered_map m(cuda::std::begin(arr), cuda::std::end(arr), 42, cuda::std::hash<short>(), cuda::std::equal_to<>(), test_allocator<PC>(0, 41));
    ASSERT_SAME_TYPE(decltype(m), cuda::std::unordered_map<int, long, cuda::std::hash<short>, cuda::std::equal_to<>, test_allocator<PC>>);
    assert(cuda::std::is_permutation(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    assert(m.get_allocator().get_id() == 41);
    }

    {
    cuda::std::unordered_map<int, long> source;
    cuda::std::unordered_map m(source);
    ASSERT_SAME_TYPE(decltype(m), decltype(source));
    assert(m.size() == 0);
    }

    {
    cuda::std::unordered_map<int, long> source;
    cuda::std::unordered_map m{source};  // braces instead of parens
    ASSERT_SAME_TYPE(decltype(m), decltype(source));
    assert(m.size() == 0);
    }

    {
    cuda::std::unordered_map<int, long, cuda::std::hash<short>, cuda::std::equal_to<>, test_allocator<PC>> source;
    test_allocator<PC> a(0, 42);
    cuda::std::unordered_map m(source, a);
    ASSERT_SAME_TYPE(decltype(m), decltype(source));
    assert(m.get_allocator().get_id() == 42);
    assert(m.size() == 0);
    }

    {
    cuda::std::unordered_map<int, long, cuda::std::hash<short>, cuda::std::equal_to<>, test_allocator<PC>> source;
    test_allocator<PC> a(0, 43);
    cuda::std::unordered_map m{source, a};  // braces instead of parens
    ASSERT_SAME_TYPE(decltype(m), decltype(source));
    assert(m.get_allocator().get_id() == 43);
    assert(m.size() == 0);
    }

    {
    cuda::std::unordered_map m { P{1,1L}, P{2,2L}, P{1,1L}, P{INT_MAX,1L}, P{3,1L} };
    ASSERT_SAME_TYPE(decltype(m), cuda::std::unordered_map<int, long>);
    assert(cuda::std::is_permutation(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    }

    {
    cuda::std::unordered_map m({ P{1,1L}, P{2,2L}, P{1,1L}, P{INT_MAX,1L}, P{3,1L} }, 42);
    ASSERT_SAME_TYPE(decltype(m), cuda::std::unordered_map<int, long>);
    assert(cuda::std::is_permutation(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    }

    {
    cuda::std::unordered_map m({ P{1,1L}, P{2,2L}, P{1,1L}, P{INT_MAX,1L}, P{3,1L} }, 42, cuda::std::hash<short>());
    ASSERT_SAME_TYPE(decltype(m), cuda::std::unordered_map<int, long, cuda::std::hash<short>>);
    assert(cuda::std::is_permutation(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    }

    {
    cuda::std::unordered_map m({ P{1,1L}, P{2,2L}, P{1,1L}, P{INT_MAX,1L}, P{3,1L} }, 42, cuda::std::hash<short>(), cuda::std::equal_to<>());
    ASSERT_SAME_TYPE(decltype(m), cuda::std::unordered_map<int, long, cuda::std::hash<short>, cuda::std::equal_to<>>);
    assert(cuda::std::is_permutation(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    }

    {
    cuda::std::unordered_map m({ P{1,1L}, P{2,2L}, P{1,1L}, P{INT_MAX,1L}, P{3,1L} }, 42, cuda::std::hash<short>(), cuda::std::equal_to<>(), test_allocator<PC>(0, 44));
    ASSERT_SAME_TYPE(decltype(m), cuda::std::unordered_map<int, long, cuda::std::hash<short>, cuda::std::equal_to<>, test_allocator<PC>>);
    assert(cuda::std::is_permutation(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    assert(m.get_allocator().get_id() == 44);
    }

    {
    const P arr[] = { {1,1}, {2,2}, {1,1}, {INT_MAX,1}, {3,1} };
    cuda::std::unordered_map m(cuda::std::begin(arr), cuda::std::end(arr), 42, test_allocator<PC>(0, 45));
    ASSERT_SAME_TYPE(decltype(m), cuda::std::unordered_map<int, long, cuda::std::hash<int>, cuda::std::equal_to<int>, test_allocator<PC>>);
    assert(cuda::std::is_permutation(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    assert(m.get_allocator().get_id() == 45);
    }

    {
    const P arr[] = { {1,1}, {2,2}, {1,1}, {INT_MAX,1}, {3,1} };
    cuda::std::unordered_map m(cuda::std::begin(arr), cuda::std::end(arr), 42, cuda::std::hash<short>(), test_allocator<PC>(0, 46));
    ASSERT_SAME_TYPE(decltype(m), cuda::std::unordered_map<int, long, cuda::std::hash<short>, cuda::std::equal_to<int>, test_allocator<PC>>);
    assert(cuda::std::is_permutation(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    assert(m.get_allocator().get_id() == 46);
    }

    {
    cuda::std::unordered_map m({ P{1,1L}, P{2,2L}, P{1,1L}, P{INT_MAX,1L}, P{3,1L} }, 42, test_allocator<PC>(0, 47));
    ASSERT_SAME_TYPE(decltype(m), cuda::std::unordered_map<int, long, cuda::std::hash<int>, cuda::std::equal_to<int>, test_allocator<PC>>);
    assert(cuda::std::is_permutation(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    assert(m.get_allocator().get_id() == 47);
    }

    {
    cuda::std::unordered_map m({ P{1,1L}, P{2,2L}, P{1,1L}, P{INT_MAX,1L}, P{3,1L} }, 42, cuda::std::hash<short>(), test_allocator<PC>(0, 48));
    ASSERT_SAME_TYPE(decltype(m), cuda::std::unordered_map<int, long, cuda::std::hash<short>, cuda::std::equal_to<int>, test_allocator<PC>>);
    assert(cuda::std::is_permutation(m.begin(), m.end(), cuda::std::begin(expected_m), cuda::std::end(expected_m)));
    assert(m.get_allocator().get_id() == 48);
    }

    return 0;
}
