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
  static_assert(cuda::pmr::detail::is_resource_pointer_convertible<derived2*, derived1*>::value,
                "A pointer to a derived class should be convertible to a pointer of public a base class");

  static_assert(cuda::pmr::detail::is_resource_pointer_convertible<derived2*,cuda::pmr::memory_resource<cuda::pmr::memory_kind::managed>*>::value,
                "A pointer to a derived class should be convertible to a pointer of public a base class");

  static_assert(cuda::pmr::detail::is_resource_pointer_convertible<derived2*, cuda::pmr::detail::memory_resource_base*>::value,
                "A pointer to a derived memory resource should be convertible to a pointer to the common base.");

  static_assert(cuda::pmr::detail::is_resource_pointer_convertible<derived_async2*, derived_async1*>::value,
                "A pointer to a derived class should be convertible to a pointer of public a base class");

  static_assert(cuda::pmr::detail::is_resource_pointer_convertible<derived_async2*, cuda::pmr::stream_ordered_resource<cuda::pmr::memory_kind::managed>*>::value,
                "A pointer to a derived class should be convertible to a pointer of public a base class");

  static_assert(cuda::pmr::detail::is_resource_pointer_convertible<derived_async2*, cuda::pmr::detail::stream_ordered_resource_base*>::value,
                "A pointer to a derived memory resource should be convertible to a pointer to the common base.");


  static_assert(!cuda::pmr::detail::is_resource_pointer_convertible<derived_async2*, derived2*>::value,
                "Pointers to unrelated classes should not be convertible");

  static_assert(!cuda::pmr::detail::is_resource_pointer_convertible<derived_async1*, derived1*>::value,
                "Pointers to unrelated classes should not be convertible");

  static_assert(!cuda::pmr::detail::is_resource_pointer_convertible<derived_async1*, derived_async2*>::value,
                "Conversion to a subclass pointer should not be possible");

  static_assert(!cuda::pmr::detail::is_resource_pointer_convertible<derived2*, cuda::pmr::stream_ordered_resource<cuda::pmr::memory_kind::managed>*>::value,
                "A pointer to a synchronous resource should not be convertible to a stream ordered one");

  static_assert(!cuda::pmr::detail::is_resource_pointer_convertible<derived2*, cuda::pmr::detail::stream_ordered_resource_base*>::value,
                "A pointer to a synchronous resource should not be convertible to a stream ordered one");

  static_assert(!cuda::pmr::detail::is_resource_pointer_convertible<cuda::pmr::detail::stream_ordered_resource_base*, cuda::pmr::stream_ordered_resource<cuda::pmr::memory_kind::managed>*>::value,
                "A pointer to a commnon base class should not be convertible to a pointer to a kind-qualified resource.");

  static_assert(!cuda::pmr::detail::is_resource_pointer_convertible<cuda::pmr::detail::memory_resource_base*, cuda::pmr::memory_resource<cuda::pmr::memory_kind::managed>*>::value,
                "A pointer to a commnon base class should not be convertible to a pointer to a kind-qualified resource.");
#endif
  return 0;
}
