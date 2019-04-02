//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <cuda/std/compare>

// class weak_equality


#include <cuda/std/compare>
#include <cuda/std/cassert>
#include "test_macros.h"

const volatile void* volatile sink;

void test_static_members() {
  DoNotOptimize(&cuda::std::weak_equality::equivalent);
  DoNotOptimize(&cuda::std::weak_equality::nonequivalent);
}

void test_signatures() {
  auto& Eq = cuda::std::weak_equality::equivalent;

  ASSERT_NOEXCEPT(Eq == 0);
  ASSERT_NOEXCEPT(0 == Eq);
  ASSERT_NOEXCEPT(Eq != 0);
  ASSERT_NOEXCEPT(0 != Eq);
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  ASSERT_NOEXCEPT(0 <=> Eq);
  ASSERT_NOEXCEPT(Eq <=> 0);
  ASSERT_SAME_TYPE(decltype(Eq <=> 0), cuda::std::weak_equality);
  ASSERT_SAME_TYPE(decltype(0 <=> Eq), cuda::std::weak_equality);
#endif
}

constexpr bool test_constexpr() {
  auto& Eq = cuda::std::weak_equality::equivalent;
  auto& NEq = cuda::std::weak_equality::nonequivalent;
  assert((Eq == 0) == true);
  assert((0 == Eq) == true);
  assert((NEq == 0) == false);
  assert((0 == NEq) == false);

  assert((Eq != 0) == false);
  assert((0 != Eq) == false);
  assert((NEq != 0) == true);
  assert((0 != NEq) == true);

#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  cuda::std::weak_equality res = (Eq <=> 0);
  ((void)res);
  res = (0 <=> Eq);
  ((void)res);
#endif

  return true;
}

int main(int, char**) {
  test_static_members();
  test_signatures();
  static_assert(test_constexpr(), "constexpr test failed");

  return 0;
}
