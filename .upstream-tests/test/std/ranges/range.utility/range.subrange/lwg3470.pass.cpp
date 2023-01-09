//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// class cuda::std::ranges::subrange;
//   Test the example from LWG 3470,
//   qualification conversions in __convertible_to_non_slicing

#include <cuda/std/ranges>

#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  // The example from LWG3470, using implicit conversion.
  int a[3] = { 1, 2, 3 };
  int* b[3] = { &a[2], &a[0], &a[1] };
  cuda::std::ranges::subrange<const int* const*> c = b;
  assert(c.begin() == b + 0);
  assert(c.end() == b + 3);

  // Also test CTAD and a subrange-to-subrange conversion.
#if TEST_STD_VER > 17
  cuda::std::same_as<cuda::std::ranges::subrange<int**>> auto d = cuda::std::ranges::subrange(b);
#else
  auto d = cuda::std::ranges::subrange<int**>(b);
#endif
  assert(d.begin() == b + 0);
  assert(d.end() == b + 3);

  cuda::std::ranges::subrange<const int* const*> e = d;
  assert(e.begin() == b + 0);
  assert(e.end() == b + 3);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
