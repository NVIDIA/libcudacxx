//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// constexpr void value() const &;
// constexpr void value() &&;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test() {
  // const &
  {
    const cuda::std::expected<void, int> e;
    e.value();
    static_assert(cuda::std::is_same_v<decltype(e.value()), void>, "");
  }

  // &
  {
    cuda::std::expected<void, int> e;
    e.value();
    static_assert(cuda::std::is_same_v<decltype(e.value()), void>, "");
  }

  // &&
  {
    cuda::std::expected<void, int> e;
    cuda::std::move(e).value();
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(e).value()), void>, "");
  }

  // const &&
  {
    const cuda::std::expected<void, int> e;
    cuda::std::move(e).value();
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(e).value()), void>, "");
  }

  return true;
}
#ifndef TEST_HAS_NO_EXCEPTIONS

enum constructor_used : unsigned int {
  default_constructed      = 1 << 0,
  lvalue_constructed       = 1 << 1,
  const_lvalue_constructed = 1 << 2,
  rvalue_constructed       = 1 << 3,
  const_rvalue_constructed = 1 << 4,
};

struct track_construction {
  track_construction() = default;
  track_construction& operator=(const track_construction&) = default;
  track_construction& operator=(track_construction&&) = default;

  __host__ __device__ track_construction(track_construction& other) noexcept
    : val_(other.val_ | lvalue_constructed) {}
  __host__ __device__ track_construction(const track_construction& other) noexcept
    : val_(other.val_ | const_lvalue_constructed) {}
  __host__ __device__ track_construction(track_construction&& other) noexcept
    : val_(other.val_ | rvalue_constructed) {}
  __host__ __device__ track_construction(const track_construction&& other) noexcept
    : val_(other.val_ | const_rvalue_constructed) {}

  unsigned int val_ = default_constructed;
};

__host__ __device__ void testException() {
  { // &
    try {
      cuda::std::expected<void, track_construction> e(cuda::std::unexpect);
      (void) e.value();
      assert(false);
    } catch (const cuda::std::bad_expected_access<int>& ex) {
      // We bind against the `const &` overload
      assert(ex.error().val_ == const_lvalue_constructed | rvalue_constructed);
    }
  }

  { // const &
    try {
      cuda::std::expected<void, track_construction> e(cuda::std::unexpect);
      (void) cuda::std::as_const(e).value();
      assert(false);
    } catch (const cuda::std::bad_expected_access<int>& ex) {
      assert(ex.error().val_ == const_lvalue_constructed | rvalue_constructed);
    }
  }

  { // &&
    try {
      cuda::std::expected<void, track_construction> e(cuda::std::unexpect);
      (void) cuda::std::move(e).value();
      assert(false);
    } catch (const cuda::std::bad_expected_access<int>& ex) {
      assert(ex.error().val_ == rvalue_constructed);
    }
  }

  { // const &&
    try {
      cuda::std::expected<void, track_construction> e(cuda::std::unexpect);
      (void) cuda::std::move(cuda::std::as_const(e)).value();
      assert(false);
    } catch (const cuda::std::bad_expected_access<int>& ex) {
      // We bind against the const& overload
      assert(ex.error().val_ == const_lvalue_constructed | rvalue_constructed);
    }
  }

  // MoveOnly
  {
    cuda::std::expected<void, MoveOnly> e(cuda::std::unexpect, 5);
    try {
      (void) cuda::std::move(e).value();
      assert(false);
    } catch (const cuda::std::bad_expected_access<MoveOnly>& ex) {
      assert(ex.error() == 5);
    }
  }
}
#endif // TEST_HAS_NO_EXCEPTIONS

int main(int, char**) {
  test();
  static_assert(test(), "");
#ifndef TEST_HAS_NO_EXCEPTIONS
  testException();
#endif // TEST_HAS_NO_EXCEPTIONS

  return 0;
}
