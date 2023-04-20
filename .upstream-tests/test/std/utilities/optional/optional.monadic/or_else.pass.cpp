//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <cuda/std/optional>

// template<class F> constexpr optional or_else(F&&) &&;
// template<class F> constexpr optional or_else(F&&) const&;

#include "MoveOnly.h"

#include <cuda/std/cassert>
#include <cuda/std/optional>

struct NonMovable {
  NonMovable() = default;
  NonMovable(NonMovable&&) = delete;
};

#if TEST_STD_VER > 17

template <class Opt, class F>
concept has_or_else = requires(Opt&& opt, F&& f) {
  {cuda::std::forward<Opt>(opt).or_else(cuda::std::forward<F>(f))};
};

#else

template <class Opt, class F>
_LIBCUDACXX_CONCEPT_FRAGMENT(
    HasOrElse,
    requires(Opt&& opt, F&& f) //
    (cuda::std::forward<Opt>(opt).or_else(cuda::std::forward<F>(f))));

template <class Opt, class F>
_LIBCUDACXX_CONCEPT has_or_else = _LIBCUDACXX_FRAGMENT(HasOrElse, Opt, F);

#endif

template <class T>
__host__ __device__
cuda::std::optional<T> return_optional();

static_assert(has_or_else<cuda::std::optional<int>&, decltype(return_optional<int>)>, "");
static_assert(has_or_else<cuda::std::optional<int>&&, decltype(return_optional<int>)>, "");
static_assert(!has_or_else<cuda::std::optional<MoveOnly>&, decltype(return_optional<MoveOnly>)>, "");
static_assert(has_or_else<cuda::std::optional<MoveOnly>&&, decltype(return_optional<MoveOnly>)>, "");
// The following cases appear to be causing GCC, specifically GCC 9, to instantiate too much and fail to sfinae in the "concept" above,
// but only in C++14. This appears to be a compiler bug present specifically in this version, but since it's failing to sfinae on an error,
// it appears that it is correctly rejecting those cases, so we are fine.
#if !(defined(TEST_COMPILER_GCC) && __GNUC__ == 9 && TEST_STD_VER == 14)
static_assert(!has_or_else<cuda::std::optional<NonMovable>&, decltype(return_optional<NonMovable>)>, "");
static_assert(!has_or_else<cuda::std::optional<NonMovable>&&, decltype(return_optional<NonMovable>)>, "");
#endif

__host__ __device__
cuda::std::optional<int> take_int(int) { return 0; }
__host__ __device__
void take_int_return_void(int) {}

static_assert(!has_or_else<cuda::std::optional<int>, decltype(take_int)>, "");
static_assert(!has_or_else<cuda::std::optional<int>, decltype(take_int_return_void)>, "");
static_assert(!has_or_else<cuda::std::optional<int>, int>, "");

__host__ __device__
TEST_CONSTEXPR_CXX17 bool test() {
  {
    cuda::std::optional<int> opt;
    assert(opt.or_else([] { return cuda::std::optional<int>{0}; }) == 0);
    opt = 1;
    opt.or_else([] {
      assert(false);
      return cuda::std::optional<int>{};
    });
  }

  return true;
}

__host__ __device__
TEST_CONSTEXPR_CXX17 bool test_nontrivial() {
  {
    cuda::std::optional<MoveOnly> opt;
    opt = cuda::std::move(opt).or_else([] { return cuda::std::optional<MoveOnly>{MoveOnly{}}; });
    cuda::std::move(opt).or_else([] {
      assert(false);
      return cuda::std::optional<MoveOnly>{};
    });
  }

  return true;
}

int main(int, char**) {
  test();
  test_nontrivial();
#if TEST_STD_VER > 17
  static_assert(test(), "");
#if defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test_nontrivial());
#endif
#endif
  return 0;
}
