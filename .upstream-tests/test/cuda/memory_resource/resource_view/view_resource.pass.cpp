//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cuda/memory_resource>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/stream_view>
#include <memory>
#include <tuple>
#include <vector>
#include "resource_hierarchy.h"

int main(int argc, char **argv) {
#ifndef __CUDA_ARCH__
  derived_async2 *res = nullptr;
  using expected_view_type = cuda::basic_resource_view<derived_async2*, cuda::is_kind<cuda::memory_kind::managed>>;
  auto view = cuda::view_resource<cuda::is_kind<cuda::memory_kind::managed>>(res);
  static_assert(std::is_same<decltype(view), expected_view_type>::value,
                "Unexpected return type of `view_resource` - expected a basic_resource_view with is_kind and concrete pointer type");
  view_DA2 vda2(res);
  vda2 = view;
  view_DA1 vda1(res);
  vda1 = view;
  vda1 = vda2;
  view_MA vma(res);
  vma = view;
  vma = vda2;
  vma = vda1;
  view_BA vba(res);
  vba = view;
  vba = vda1;
  vba = vma;
  vba = vba;
  view_M vm(res);
  vm = view;
  vm = vda1;
  vm = vma;
  vm = vm;
  view_B vb(res);
  vb = view;
  vb = vda1;
  vb = vma;
  vb = vba;
  vb = vm;
#endif
  return 0;
}
