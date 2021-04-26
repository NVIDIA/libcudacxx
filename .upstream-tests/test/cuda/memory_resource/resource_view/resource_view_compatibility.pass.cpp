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
  static_assert(cuda::is_view_convertible<view_D2, view_D2>::value,
                "A resource view should be convertible to a self.");

  static_assert(cuda::is_view_convertible<view_D2, view_D1>::value,
                "A resource view should be convertible to a view of a superclass with a subset of properties");

  static_assert(cuda::is_view_convertible<view_D2, view_M>::value,
                "A resource view should be convertible to a view of a superclass with a subset of properties");

  static_assert(cuda::is_view_convertible<view_D2, view_B>::value,
                "A resource view should be convertible to a view of common base with a subset of properties");

  static_assert(cuda::is_view_convertible<view_D1, view_M>::value,
                "A resource view should be convertible to a view of a superclass with a subset of properties");

  static_assert(cuda::is_view_convertible<view_D1, view_B>::value,
                "A resource view should be convertible to a view of common base with a subset of properties");

  static_assert(cuda::is_view_convertible<view_M, view_M>::value,
                "A resource view should be convertible to self.");

  static_assert(cuda::is_view_convertible<view_M, view_B>::value,
                "A resource view should be convertible to a view of common base with a subset of properties");

  static_assert(cuda::is_view_convertible<view_B, view_B>::value,
                "A resource view should be convertible to self.");

  static_assert(cuda::is_view_convertible<view_DA2, view_DA1>::value,
                "A resource view should be convertible to a view of a superclass with a subset of properties");

  static_assert(cuda::is_view_convertible<view_DA2, view_MA>::value,
                "A resource view should be convertible to a view of a superclass with a subset of properties");

  static_assert(cuda::is_view_convertible<view_DA2, view_M>::value,
                "A resource view should be convertible to a view of a superclass with a subset of properties");

  static_assert(cuda::is_view_convertible<view_DA2, view_BA>::value,
                "A resource view should be convertible to a view of common base with a subset of properties");

  static_assert(cuda::is_view_convertible<view_DA2, view_B>::value,
                "A resource view should be convertible to a view of common base with a subset of properties");


  static_assert(cuda::is_view_convertible<view_DA1, view_DA1>::value,
                "A resource view should be convertible to self.");

  static_assert(cuda::is_view_convertible<view_DA1, view_MA>::value,
                "A resource view should be convertible to a view of a superclass with a subset of properties");

  static_assert(cuda::is_view_convertible<view_DA1, view_BA>::value,
                "A resource view should be convertible to a view of common base with a subset of properties");

  static_assert(cuda::is_view_convertible<view_MA, view_MA>::value,
                "A resource view should be convertible to self.");

  static_assert(cuda::is_view_convertible<view_MA, view_M>::value,
                "A resource view should be convertible to a view of a superclass with a subset of properties");

  static_assert(cuda::is_view_convertible<view_MA, view_B>::value,
                "A resource view should be convertible to a view of common base with a subset of properties");

  static_assert(cuda::is_view_convertible<view_BA, view_BA>::value,
                "A resource view should be convertible to self.");

  static_assert(cuda::is_view_convertible<view_BA, view_B>::value,
                "A resource view should be convertible to a view of common base with a subset of properties");


  static_assert(!cuda::is_view_convertible<view_DA2, view_D2>::value,
                "Views to unrelated types should not be compatible.");

  static_assert(!cuda::is_view_convertible<view_D1, view_DA1>::value,
                "Views to unrelated types should not be compatible.");

  static_assert(!cuda::is_view_convertible<view_DA1, view_DA2>::value,
                "A resource view should not to a view with a superclass pointer.");

  static_assert(!cuda::is_view_convertible<view_MA, view_DA1>::value,
                "A resource view should not to a view with a superclass pointer.");

  static_assert(!cuda::is_view_convertible<view_D1, view_D2>::value,
                "A resource view should not to a view with a superclass pointer.");

  static_assert(!cuda::is_view_convertible<view_M, view_D1>::value,
                "A resource view should not to a view with a superclass pointer.");

  static_assert(!cuda::is_view_convertible<view_B, view_M>::value,
                "A resource view should not to a view with a superclass pointer.");

  static_assert(!cuda::is_view_convertible<view_B, view_BA>::value,
                "A resource view to a syncrhouns resource cannot be converted to a stream ordered one.");

  static_assert(!cuda::is_view_convertible<view_M, view_MA>::value,
                "A resource view to a syncrhouns resource cannot be converted to a stream ordered one.");

  return 0;
}
