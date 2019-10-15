//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <cuda/std/compare>

// template <class ...Ts> struct common_comparison_category
// template <class ...Ts> using common_comparison_category_t


#include <cuda/std/compare>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

const volatile void* volatile sink;

template <class Expect, class ...Args>
void test_cat() {
  using Cat = cuda::std::common_comparison_category<Args...>;
  using CatT = typename Cat::type;
  static_assert(cuda::std::is_same<CatT, cuda::std::common_comparison_category_t<Args...>>::value, "");
  static_assert(cuda::std::is_same<CatT, Expect>::value, "expected different category");
};


// [class.spaceship]p4: The 'common comparison type' U of a possibly-empty list
//   of 'n' types T0, T1, ..., TN, is defined as follows:
int main(int, char**) {
  using WE = cuda::std::weak_equality;
  using SE = cuda::std::strong_equality;
  using PO = cuda::std::partial_ordering;
  using WO = cuda::std::weak_ordering;
  using SO = cuda::std::strong_ordering;

  // [class.spaceship]p4.1: If any Ti is not a comparison category tpe, U is void.
  {
    test_cat<void, void>();
    test_cat<void, int*>();
    test_cat<void, SO&>();
    test_cat<void, SO const>();
    test_cat<void, SO*>();
    test_cat<void, SO, void, SO>();
  }

  // [class.spaceship]p4.2: Otherwise, if at least on Ti is
  // cuda::std::weak_equality, or at least one Ti is cuda::std::strong_equality and at least
  // one Tj is cuda::std::partial_ordering or cuda::std::weak_ordering, U is cuda::std::weak_equality
  {
    test_cat<WE, WE>();
    test_cat<WE, SO, WE, SO>();
    test_cat<WE, SE, SO, PO>();
    test_cat<WE, WO, SO, SE>();
  }

  // [class.spaceship]p4.3: Otherwise, if at least one Ti is cuda::std::strong_equality,
  // U is cuda::std::strong_equality
  {
    test_cat<SE, SE>();
    test_cat<SE, SO, SE, SO>();
  }

  // [class.spaceship]p4.4: Otherwise, if at least one Ti is cuda::std::partial_ordering,
  // U is cuda::std::partial_ordering
  {
    test_cat<PO, PO>();
    test_cat<PO, SO, PO, SO>();
    test_cat<PO, WO, PO, SO>();
  }

  // [class.spaceship]p4.5: Otherwise, if at least one Ti is cuda::std::weak_ordering,
  // U is cuda::std::weak_ordering
  {
    test_cat<WO, WO>();
    test_cat<WO, SO, WO, SO>();
  }

  // [class.spaceship]p4.6: Otherwise, U is cuda::std::strong_ordering. [Note: in
  // particular this is the result when n is 0. -- end note]
  {
    test_cat<SO>(); // empty type list
    test_cat<SO, SO>();
    test_cat<SO, SO, SO>();
  }

  return 0;
}
