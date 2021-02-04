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
#include <cuda/std/type_traits>
#include <cuda/stream_view>

template <cuda::memory_kind Kind> constexpr bool test_memory_kind() {
  using mr = cuda::stream_ordered_memory_resource<Kind>;
  return mr::kind == Kind;
}

template <cuda::memory_kind Kind, std::size_t Alignment>
constexpr bool test_alignment() {
  using mr = cuda::stream_ordered_memory_resource<Kind>;
  return mr::default_alignment == Alignment;
}

int main(int argc, char **argv) {

#ifndef __CUDA_ARCH__
  using cuda::memory_kind;
  static_assert(test_memory_kind<memory_kind::host>(), "");
  static_assert(test_memory_kind<memory_kind::device>(), "");
  static_assert(test_memory_kind<memory_kind::unified>(), "");
  static_assert(test_memory_kind<memory_kind::pinned>(), "");

  using mr = cuda::stream_ordered_memory_resource<memory_kind::host>;
  static_assert(cuda::std::is_same<mr::context, cuda::any_context>::value, "");

  static_assert(test_alignment<memory_kind::host, alignof(cuda::std::max_align_t)>(), "");
  static_assert(test_alignment<memory_kind::device, alignof(cuda::std::max_align_t)>(), "");
  static_assert(test_alignment<memory_kind::unified, alignof(cuda::std::max_align_t)>(), "");
  static_assert(test_alignment<memory_kind::pinned, alignof(cuda::std::max_align_t)>(), "");
#endif

  return 0;
}
