//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <cuda/std/compare>

// class strong_equality


#include <cuda/std/compare>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

const volatile void* volatile sink;

void test_static_members() {
  DoNotOptimize(&cuda::std::strong_equality::equal);
  DoNotOptimize(&cuda::std::strong_equality::nonequal);
  DoNotOptimize(&cuda::std::strong_equality::equivalent);
  DoNotOptimize(&cuda::std::strong_equality::nonequivalent);
}

void test_signatures() {
  auto& Eq = cuda::std::strong_equality::equivalent;

  ASSERT_NOEXCEPT(Eq == 0);
  ASSERT_NOEXCEPT(0 == Eq);
  ASSERT_NOEXCEPT(Eq != 0);
  ASSERT_NOEXCEPT(0 != Eq);
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  ASSERT_NOEXCEPT(0 <=> Eq);
  ASSERT_NOEXCEPT(Eq <=> 0);
  ASSERT_SAME_TYPE(decltype(Eq <=> 0), cuda::std::strong_equality);
  ASSERT_SAME_TYPE(decltype(0 <=> Eq), cuda::std::strong_equality);
#endif
}

void test_conversion() {
  constexpr cuda::std::weak_equality res = cuda::std::strong_equality::equivalent;
  static_assert(res == 0, "");
  static_assert(cuda::std::is_convertible<const cuda::std::strong_equality&,
      cuda::std::weak_equality>::value, "");
  static_assert(res == 0, "expected equal");

  constexpr cuda::std::weak_equality neq_res = cuda::std::strong_equality::nonequivalent;
  static_assert(neq_res != 0, "expected not equal");
}

constexpr bool test_constexpr() {
  auto& Eq = cuda::std::strong_equality::equal;
  auto& NEq = cuda::std::strong_equality::nonequal;
  auto& Equiv = cuda::std::strong_equality::equivalent;
  auto& NEquiv = cuda::std::strong_equality::nonequivalent;
  assert((Eq == 0) == true);
  assert((0 == Eq) == true);
  assert((Equiv == 0) == true);
  assert((0 == Equiv) == true);
  assert((NEq == 0) == false);
  assert((0 == NEq) == false);
  assert((NEquiv == 0) == false);
  assert((0 == NEquiv) == false);

  assert((Eq != 0) == false);
  assert((0 != Eq) == false);
  assert((Equiv != 0) == false);
  assert((0 != Equiv) == false);
  assert((NEq != 0) == true);
  assert((0 != NEq) == true);
  assert((NEquiv != 0) == true);
  assert((0 != NEquiv) == true);

#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  cuda::std::strong_equality res = (Eq <=> 0);
  ((void)res);
  res = (0 <=> Eq);
  ((void)res);
#endif

  return true;
}

int main(int, char**) {
  test_static_members();
  test_signatures();
  test_conversion();
  static_assert(test_constexpr(), "constexpr test failed");

  return 0;
}
