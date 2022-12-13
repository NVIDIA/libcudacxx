//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: gcc-8, gcc-9

// template<class F, class... Args>
// concept predicate;

#include <cuda/std/concepts>

#include "test_macros.h"
#if TEST_STD_VER > 17

__host__ __device__ constexpr bool check_subsumption(cuda::std::regular_invocable auto) {
  return false;
}

// clang-format off
template<class F>
requires cuda::std::predicate<F> && true
__host__ __device__ constexpr bool check_subsumption(F)
{
  return true;
}
// clang-format on

static_assert(!check_subsumption([] {}), "");
static_assert(check_subsumption([] { return true; }), "");

#endif // TEST_STD_VER > 17

int main(int, char**)
{
  return 0;
}
