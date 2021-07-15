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
  {
    view_D2 derived2_view;
    derived2_view = derived2_view;

    view_D1 derived1_view = derived2_view;
    derived1_view = derived2_view;
    derived1_view = derived1_view;


    view_M memres_view = derived2_view;
    memres_view = derived2_view;
    memres_view = derived1_view;
    memres_view = memres_view;

    view_B base_view = derived2_view;

    base_view = derived2_view;
    base_view = derived1_view;
    base_view = memres_view;
    base_view = base_view;
  }

  {

    view_DA2 derived_async2_view;
    derived_async2_view = derived_async2_view;

    view_DA1 derived_async1_view = derived_async2_view;
    derived_async1_view = derived_async2_view;
    derived_async1_view = derived_async1_view;

    view_MA memres_async_view = derived_async2_view;
    memres_async_view = derived_async2_view;
    memres_async_view = derived_async1_view;
    memres_async_view = memres_async_view;

    view_M memres_view = derived_async2_view;
    memres_view = derived_async2_view;
    memres_view = derived_async1_view;
    memres_view = memres_async_view;
    memres_view = memres_view;

    view_BA base_async_view = derived_async2_view;
    base_async_view = derived_async2_view;
    base_async_view = derived_async1_view;
    base_async_view = memres_async_view;

    view_B base_view = derived_async2_view;
    base_view = derived_async2_view;
    base_view = derived_async1_view;
    base_view = memres_async_view;
    base_view = memres_view;
    base_view = base_async_view;
  }
#endif
  return 0;
}
