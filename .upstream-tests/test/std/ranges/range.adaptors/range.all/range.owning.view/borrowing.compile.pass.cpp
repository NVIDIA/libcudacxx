//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class T>
//   inline constexpr bool enable_borrowed_range<owning_view<T>> = enable_borrowed_range<T>;

#include <cuda/std/ranges>

#include "test_range.h"

static_assert( cuda::std::ranges::borrowed_range<cuda::std::ranges::owning_view<BorrowedView>>);
static_assert(!cuda::std::ranges::borrowed_range<cuda::std::ranges::owning_view<NonBorrowedView>>);

int main(int, char**) {
  return 0;
}
