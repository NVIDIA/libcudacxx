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


class sync_resource : public cuda::memory_resource<cuda::memory_kind::pinned> {
public:
  int extra_sync() {
    return 42;
  }

  size_t allocated_size = 0, allocated_alignment = 0;
  void *allocated_pointer = nullptr;
  mutable const cuda::memory_resource<cuda::memory_kind::pinned> *compared_resource = nullptr;
private:
  void *do_allocate(size_t size, size_t alignment) override {
    allocated_size = size;
    allocated_alignment = alignment;
    return allocated_pointer = reinterpret_cast<void*>(0x123400);
  }

  void do_deallocate(void *mem, size_t size, size_t alignment) {
    assert(mem == allocated_pointer);
    assert(size == allocated_size);
    assert(alignment == allocated_alignment);
    allocated_pointer = 0;
    allocated_size = 0;
    allocated_alignment = 0;
  }

  bool do_is_equal(const cuda::memory_resource<cuda::memory_kind::pinned> &other) const noexcept override {
    compared_resource = &other;
    return this == &other;
  }
};

class async_resource : public cuda::stream_ordered_memory_resource<cuda::memory_kind::device> {
public:
  int extra_async() {
    return 42;
  }

  size_t allocated_size = 0, allocated_alignment = 0;
  void *allocated_pointer = nullptr;
  cuda::stream_view allocation_stream = {};
  mutable const cuda::memory_resource<cuda::memory_kind::device> *compared_resource = nullptr;
private:
  void *do_allocate(size_t size, size_t alignment) override {
    allocated_size = size;
    allocated_alignment = alignment;
    return allocated_pointer = reinterpret_cast<void*>(0x123400);
  }

  void *do_allocate_async(size_t size, size_t alignment, cuda::stream_view stream) override {
    allocated_size = size;
    allocated_alignment = alignment;
    allocation_stream = stream;
    return allocated_pointer = reinterpret_cast<void*>(0x123400);
  }

  void do_deallocate(void *mem, size_t size, size_t alignment) {
    assert(mem == allocated_pointer);
    assert(size == allocated_size);
    assert(alignment == allocated_alignment);
    allocated_pointer = 0;
    allocated_size = 0;
    allocated_alignment = 0;
  }

  void do_deallocate_async(void *mem, size_t size, size_t alignment, cuda::stream_view stream) override {
    assert(mem == allocated_pointer);
    assert(size == allocated_size);
    assert(stream == allocation_stream);
    assert(alignment == allocated_alignment);
    assert(stream == allocation_stream);
    allocated_pointer = 0;
    allocated_size = 0;
    allocated_alignment = 0;
    allocation_stream = {};
  }

  bool do_is_equal(const cuda::memory_resource<cuda::memory_kind::device> &other) const noexcept override {
    compared_resource = &other;
    return this == &other;
  }
};

int main(int argc, char **argv) {
#ifndef __CUDA_ARCH__
  // syncrhonous resource
  {
    sync_resource rsrc;
    auto view = cuda::view_resource<cuda::memory_access::host>(&rsrc);
    assert(view->extra_sync() == 42);
    void *ptr = view->allocate(23, 32);
    assert(ptr == view->allocated_pointer);
    assert(23 == view->allocated_size);
    assert(32 == view->allocated_alignment);
    view->deallocate(ptr, 23, 32);
    assert(nullptr == view->allocated_pointer);
    assert(0 == view->allocated_size);
    assert(0 == view->allocated_alignment);

    assert(view->is_equal(rsrc));
    assert(view->compared_resource == &rsrc);
  }
  {
    sync_resource rsrc;
    cuda::resource_view<cuda::memory_access::host> view = &rsrc;
    void *ptr = view->allocate(23, 32);
    assert(ptr == rsrc.allocated_pointer);
    assert(23 == rsrc.allocated_size);
    assert(32 == rsrc.allocated_alignment);
    view->deallocate(ptr, 23, 32);
    assert(nullptr == rsrc.allocated_pointer);
    assert(0 == rsrc.allocated_size);
    assert(0 == rsrc.allocated_alignment);
  }
  // stream-ordered resource
  {
    cuda::stream_view stream((cudaStream_t)0x1234);
    async_resource rsrc;
    auto view = cuda::view_resource<cuda::memory_access::device>(&rsrc);

    assert(view->extra_async() == 42);

    void *ptr = view->allocate(23, 32);
    assert(ptr == view->allocated_pointer);
    assert(23 == view->allocated_size);
    assert(32 == view->allocated_alignment);

    view->deallocate(ptr, 23, 32);
    assert(nullptr == view->allocated_pointer);
    assert(0 == view->allocated_size);
    assert(0 == view->allocated_alignment);
    assert(0 == view->allocated_alignment);

    ptr = view->allocate_async(42, 64, stream);
    assert(ptr == view->allocated_pointer);
    assert(42 == view->allocated_size);
    assert(64 == view->allocated_alignment);
    assert(stream.get() == view->allocation_stream.get());

    view->deallocate_async(ptr, 42, 64, stream);
    assert(nullptr == view->allocated_pointer);
    assert(0 == view->allocated_size);
    assert(0 == view->allocated_alignment);
    assert(0 == view->allocation_stream.get());

    assert(view->is_equal(rsrc));
    assert(view->compared_resource == &rsrc);
  }
  {
    cuda::stream_view stream((cudaStream_t)0x1234);
    async_resource rsrc;
    cuda::stream_ordered_resource_view<cuda::memory_access::device> view = &rsrc;

    void *ptr = view->allocate(23, 32);
    assert(ptr == rsrc.allocated_pointer);
    assert(23 == rsrc.allocated_size);
    assert(32 == rsrc.allocated_alignment);

    view->deallocate(ptr, 23, 32);
    assert(nullptr == rsrc.allocated_pointer);
    assert(0 == rsrc.allocated_size);
    assert(0 == rsrc.allocated_alignment);
    assert(0 == rsrc.allocated_alignment);

    ptr = view->allocate_async(42, 64, stream);
    assert(ptr == rsrc.allocated_pointer);
    assert(42 == rsrc.allocated_size);
    assert(64 == rsrc.allocated_alignment);
    assert(stream.get() == rsrc.allocation_stream.get());

    view->deallocate_async(ptr, 42, 64, stream);
    assert(nullptr == rsrc.allocated_pointer);
    assert(0 == rsrc.allocated_size);
    assert(0 == rsrc.allocated_alignment);
    assert(0 == rsrc.allocation_stream.get());
  }
#endif
  return 0;
}
