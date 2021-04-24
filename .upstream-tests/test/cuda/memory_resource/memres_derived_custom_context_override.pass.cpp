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

struct custom_context {};


template <typename MemoryKind>
class derived_resource : public cuda::memory_resource<MemoryKind, custom_context> {
public:
private:
  void *do_allocate(cuda::std::size_t, cuda::std::size_t) override {
    return nullptr;
  }
  void do_deallocate(void *, cuda::std::size_t, cuda::std::size_t) override {}

  custom_context do_get_context() const noexcept override { return custom_context{}; }
};

template <typename MemoryKind> void test_derived_resource() {
  using derived = derived_resource<MemoryKind>;
  static_assert(std::is_same<typename derived::context, custom_context>::value, "");
  derived d;
  auto context = d.get_context();
  static_assert(std::is_same<decltype(context), custom_context>::value, "");

  using base = cuda::memory_resource<MemoryKind, custom_context>;
  static_assert(std::is_same<typename base::context, custom_context>::value, "");
  base* b = &d;
  auto base_context = b->get_context();
  static_assert(std::is_same<decltype(base_context), custom_context>::value, "");
}


int main(int argc, char **argv) {

#ifndef __CUDA_ARCH__
  test_derived_resource<cuda::memory_kind::host>();
  test_derived_resource<cuda::memory_kind::device>();
  test_derived_resource<cuda::memory_kind::managed>();
  test_derived_resource<cuda::memory_kind::pinned>();
#endif

  return 0;
}
