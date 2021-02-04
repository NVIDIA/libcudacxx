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

struct event {
  enum action { ALLOCATE, DEALLOCATE };
  action act;
  std::uintptr_t pointer;
  cuda::std::size_t bytes;
  cuda::std::size_t alignment;
  cuda::stream_view stream;
};

bool operator==(event const &lhs, event const &rhs) {
  return std::tie(lhs.act, lhs.pointer, lhs.bytes, lhs.alignment, lhs.stream) ==
         std::tie(rhs.act, rhs.pointer, rhs.bytes, rhs.alignment, rhs.stream);
}

template <cuda::memory_kind Kind>
class derived_resource : public cuda::stream_ordered_memory_resource<Kind> {
public:
  std::vector<event> &events() { return events_; }

private:
  void *do_allocate_async(cuda::std::size_t bytes, cuda::std::size_t alignment,
                          cuda::stream_view stream) override {
    auto p = 0xDEADBEEF;
    events().push_back(event{event::ALLOCATE, p, bytes, alignment, stream});
    return reinterpret_cast<void *>(p);
  }

  void do_deallocate_async(void *p, cuda::std::size_t bytes,
                     cuda::std::size_t alignment,
                     cuda::stream_view stream) override {
    events().push_back(event{event::DEALLOCATE,
                             reinterpret_cast<std::uintptr_t>(p), bytes,
                             alignment, stream});
  }

  std::vector<event> events_;
};

template <cuda::memory_kind Kind> void test_derived_resource() {
  using derived = derived_resource<Kind>;
  using base = cuda::stream_ordered_memory_resource<Kind>;

  derived d;
  base *b = &d;

  assert(b->is_equal(*b));
  assert(b->is_equal(d));

  cuda::stream_view default_stream;

  auto p0 = b->allocate(100);
  assert(d.events().size() == 1);
  assert((d.events().back() == event{event::ALLOCATE,
                                     reinterpret_cast<std::uintptr_t>(p0), 100,
                                     derived::default_alignment, default_stream}));

  auto p1 = b->allocate(42, 32);
  assert(d.events().size() == 2);
  assert((d.events().back() == event{event::ALLOCATE,
                                     reinterpret_cast<std::uintptr_t>(p1), 42,
                                     32, default_stream}));

  b->deallocate(p0, 100);
  assert(d.events().size() == 3);
  assert((d.events().back() == event{event::DEALLOCATE,
                                     reinterpret_cast<std::uintptr_t>(p0), 100,
                                     derived::default_alignment, default_stream}));

  b->deallocate(p1, 42, 32);
  assert(d.events().size() == 4);
  assert((d.events().back() == event{event::DEALLOCATE,
                                     reinterpret_cast<std::uintptr_t>(p1), 42,
                                     32, default_stream}));

  cuda::stream_view s = reinterpret_cast<cudaStream_t>(13);

  auto p2 = b->allocate_async(123, s);
  assert(d.events().size() == 5);
  assert((d.events().back() == event{event::ALLOCATE,
                                     reinterpret_cast<std::uintptr_t>(p2), 123,
                                     derived::default_alignment, s}));

  auto p3 = b->allocate_async(42, 64, s);
  assert(d.events().size() == 6);
  assert((d.events().back() == event{event::ALLOCATE,
                                     reinterpret_cast<std::uintptr_t>(p3), 42,
                                     64, s}));

  b->deallocate_async(p2, 123, s);
  assert(d.events().size() == 7);
  assert((d.events().back() == event{event::DEALLOCATE,
                                     reinterpret_cast<std::uintptr_t>(p2), 123,
                                     derived::default_alignment, s}));

  b->deallocate_async(p3, 42, 64, s);
  assert(d.events().size() == 8);
  assert((d.events().back() == event{event::DEALLOCATE,
                                     reinterpret_cast<std::uintptr_t>(p3), 42,
                                     64, s}));
}

int main(int argc, char **argv) {

#ifndef __CUDA_ARCH__
  test_derived_resource<cuda::memory_kind::host>();
  test_derived_resource<cuda::memory_kind::device>();
  test_derived_resource<cuda::memory_kind::unified>();
  test_derived_resource<cuda::memory_kind::pinned>();
#endif

  return 0;
}
