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


/// This should fail to compile as it doesn't provide an override for `do_get_context`
template <typename MemoryKind>
class derived_resource : public cuda::memory_resource<MemoryKind, custom_context> {
public:
private:
  void *do_allocate(cuda::std::size_t, cuda::std::size_t) override {
    return nullptr;
  }
  void do_deallocate(void *, cuda::std::size_t, cuda::std::size_t) override {}
};

template <typename MemoryKind> void test_derived_resource() {
  using derived = derived_resource<MemoryKind>;
  derived d;
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
