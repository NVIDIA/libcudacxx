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
#include <memory>
#include <tuple>
#include <vector>


template <typename tag>
class resource : public cuda::memory_resource<cuda::memory_kind::pinned> {
public:
  int value = 0;
private:
  void *do_allocate(size_t, size_t) override {
    return nullptr;
  }

  void do_deallocate(void *, size_t, size_t) {
  }

#ifndef _LIBCUDACXX_NO_RTTI
  bool do_is_equal(const cuda::memory_resource<cuda::memory_kind::pinned> &other) const noexcept override {
    if (auto *other_ptr = dynamic_cast<const resource *>(&other)) {
      return value == other_ptr->value;
    } else {
      return false;
    }
  }
#endif
};


struct tag1;
struct tag2;

int main(int argc, char **argv) {
#if !defined(__CUDA_ARCH__) && !defined(_LIBCUDACXX_NO_RTTI)
  resource<tag1> r1, r2, r3;
  resource<tag2> r4;
  r1.value = 42;
  r2.value = 42;
  r3.value = 99;
  r4.value = 42;

  assert(view_resource(&r1) == view_resource(&r2));
  assert(view_resource(&r1) != view_resource(&r3));
  assert(view_resource(&r1) == view_resource(&r4));
#endif
  return 0;
}
