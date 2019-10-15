//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/set>
// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: libcpp-no-deduction-guides
// UNSUPPORTED: apple-clang-9.1

// template<class InputIterator,
//          class Compare = less<iter-value-type<InputIterator>>,
//          class Allocator = allocator<iter-value-type<InputIterator>>>
// set(InputIterator, InputIterator,
//     Compare = Compare(), Allocator = Allocator())
//   -> set<iter-value-type<InputIterator>, Compare, Allocator>;
// template<class Key, class Compare = less<Key>,
//          class Allocator = allocator<Key>>
// set(initializer_list<Key>, Compare = Compare(), Allocator = Allocator())
//   -> set<Key, Compare, Allocator>;
// template<class InputIterator, class Allocator>
// set(InputIterator, InputIterator, Allocator)
//   -> set<iter-value-type<InputIterator>,
//          less<iter-value-type<InputIterator>>, Allocator>;
// template<class Key, class Allocator>
// set(initializer_list<Key>, Allocator)
//   -> set<Key, less<Key>, Allocator>;

#include <cuda/std/algorithm> // cuda::std::equal
#include <cuda/std/cassert>
#include <cuda/std/climits> // INT_MAX
#include <cuda/std/functional>
#include <cuda/std/set>
#include <cuda/std/type_traits>

#include "test_allocator.h"

struct NotAnAllocator {
  friend bool operator<(NotAnAllocator, NotAnAllocator) { return false; }
};

int main(int, char **) {
  {
    const int arr[] = { 1, 2, 1, INT_MAX, 3 };
    cuda::std::set s(cuda::std::begin(arr), cuda::std::end(arr));

    ASSERT_SAME_TYPE(decltype(s), cuda::std::set<int>);
    const int expected_s[] = { 1, 2, 3, INT_MAX };
    assert(cuda::std::equal(s.begin(), s.end(), cuda::std::begin(expected_s),
                      cuda::std::end(expected_s)));
  }

  {
    const int arr[] = { 1, 2, 1, INT_MAX, 3 };
    cuda::std::set s(cuda::std::begin(arr), cuda::std::end(arr), cuda::std::greater<int>());

    ASSERT_SAME_TYPE(decltype(s), cuda::std::set<int, cuda::std::greater<int> >);
    const int expected_s[] = { INT_MAX, 3, 2, 1 };
    assert(cuda::std::equal(s.begin(), s.end(), cuda::std::begin(expected_s),
                      cuda::std::end(expected_s)));
  }

  {
    const int arr[] = { 1, 2, 1, INT_MAX, 3 };
    cuda::std::set s(cuda::std::begin(arr), cuda::std::end(arr), cuda::std::greater<int>(),
               test_allocator<int>(0, 42));

    ASSERT_SAME_TYPE(decltype(s),
                     cuda::std::set<int, cuda::std::greater<int>, test_allocator<int> >);
    const int expected_s[] = { INT_MAX, 3, 2, 1 };
    assert(cuda::std::equal(s.begin(), s.end(), cuda::std::begin(expected_s),
                      cuda::std::end(expected_s)));
    assert(s.get_allocator().get_id() == 42);
  }

  {
    cuda::std::set<long> source;
    cuda::std::set s(source);
    ASSERT_SAME_TYPE(decltype(s), cuda::std::set<long>);
    assert(s.size() == 0);
  }

  {
    cuda::std::set<long> source;
    cuda::std::set s{ source };  // braces instead of parens
    ASSERT_SAME_TYPE(decltype(s), cuda::std::set<long>);
    assert(s.size() == 0);
  }

  {
    cuda::std::set<long> source;
    cuda::std::set s(source, cuda::std::set<long>::allocator_type());
    ASSERT_SAME_TYPE(decltype(s), cuda::std::set<long>);
    assert(s.size() == 0);
  }

  {
    cuda::std::set s{ 1, 2, 1, INT_MAX, 3 };

    ASSERT_SAME_TYPE(decltype(s), cuda::std::set<int>);
    const int expected_s[] = { 1, 2, 3, INT_MAX };
    assert(cuda::std::equal(s.begin(), s.end(), cuda::std::begin(expected_s),
                      cuda::std::end(expected_s)));
  }

  {
    cuda::std::set s({ 1, 2, 1, INT_MAX, 3 }, cuda::std::greater<int>());

    ASSERT_SAME_TYPE(decltype(s), cuda::std::set<int, cuda::std::greater<int> >);
    const int expected_s[] = { INT_MAX, 3, 2, 1 };
    assert(cuda::std::equal(s.begin(), s.end(), cuda::std::begin(expected_s),
                      cuda::std::end(expected_s)));
  }

  {
    cuda::std::set s({ 1, 2, 1, INT_MAX, 3 }, cuda::std::greater<int>(),
               test_allocator<int>(0, 43));

    ASSERT_SAME_TYPE(decltype(s),
                     cuda::std::set<int, cuda::std::greater<int>, test_allocator<int> >);
    const int expected_s[] = { INT_MAX, 3, 2, 1 };
    assert(cuda::std::equal(s.begin(), s.end(), cuda::std::begin(expected_s),
                      cuda::std::end(expected_s)));
    assert(s.get_allocator().get_id() == 43);
  }

  {
    const int arr[] = { 1, 2, 1, INT_MAX, 3 };
    cuda::std::set s(cuda::std::begin(arr), cuda::std::end(arr), test_allocator<int>(0, 44));

    ASSERT_SAME_TYPE(decltype(s),
                     cuda::std::set<int, cuda::std::less<int>, test_allocator<int> >);
    const int expected_s[] = { 1, 2, 3, INT_MAX };
    assert(cuda::std::equal(s.begin(), s.end(), cuda::std::begin(expected_s),
                      cuda::std::end(expected_s)));
    assert(s.get_allocator().get_id() == 44);
  }

  {
    cuda::std::set s({ 1, 2, 1, INT_MAX, 3 }, test_allocator<int>(0, 45));

    ASSERT_SAME_TYPE(decltype(s),
                     cuda::std::set<int, cuda::std::less<int>, test_allocator<int> >);
    const int expected_s[] = { 1, 2, 3, INT_MAX };
    assert(cuda::std::equal(s.begin(), s.end(), cuda::std::begin(expected_s),
                      cuda::std::end(expected_s)));
    assert(s.get_allocator().get_id() == 45);
  }

  {
    NotAnAllocator a;
    cuda::std::set s{ a }; // set(initializer_list<NotAnAllocator>)
    ASSERT_SAME_TYPE(decltype(s), cuda::std::set<NotAnAllocator>);
    assert(s.size() == 1);
  }

  {
    cuda::std::set<long> source;
    cuda::std::set s{ source, source }; // set(initializer_list<set<long>>)
    ASSERT_SAME_TYPE(decltype(s), cuda::std::set<cuda::std::set<long> >);
    assert(s.size() == 1);
  }

  {
    NotAnAllocator a;
    cuda::std::set s{ a, a }; // set(initializer_list<NotAnAllocator>)
    ASSERT_SAME_TYPE(decltype(s), cuda::std::set<NotAnAllocator>);
    assert(s.size() == 1);
  }

  {
    int source[3] = { 3, 4, 5 };
    cuda::std::set s(source, source + 3); // set(InputIterator, InputIterator)
    ASSERT_SAME_TYPE(decltype(s), cuda::std::set<int>);
    assert(s.size() == 3);
  }

  {
    int source[3] = { 3, 4, 5 };
    cuda::std::set s{ source, source + 3 }; // set(initializer_list<int*>)
    ASSERT_SAME_TYPE(decltype(s), cuda::std::set<int *>);
    assert(s.size() == 2);
  }

  return 0;
}
