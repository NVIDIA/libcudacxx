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
#include <vector>
#include <memory>
#include <tuple>

struct event {
  enum action { ALLOCATE, DEALLOCATE };
  action act;
  std::uintptr_t pointer;
  cuda::std::size_t bytes;
  cuda::std::size_t alignment;
};

bool operator==(event const& lhs, event const& rhs){
  return std::tie(lhs.act, lhs.pointer, lhs.bytes, lhs.alignment) ==
         std::tie(rhs.act, rhs.pointer, rhs.bytes, rhs.alignment);
}

template <cuda::memory_kind Kind> 
class derived_resource : public cuda::memory_resource<Kind> {
public:
  std::vector<event> &events() { return events_; }
private:
  void *do_allocate(cuda::std::size_t bytes,
                    cuda::std::size_t alignment) override {
    auto p = 0xDEADBEEF;
    events().push_back(event{event::ALLOCATE, p, bytes, alignment});
    return reinterpret_cast<void*>(p);
  }

  void do_deallocate(void *p, cuda::std::size_t bytes,
                     cuda::std::size_t alignment) override {
    events().push_back(event{event::DEALLOCATE,
                             reinterpret_cast<std::uintptr_t>(p), bytes,
                             alignment});
  }

  std::vector<event> events_;
};

template <cuda::memory_kind Kind>
void test_derived_resource(){
    using derived = derived_resource<Kind>;
    using base = cuda::memory_resource<Kind>;

    derived d;
    base * b = &d;

    assert(b->is_equal(*b));
    assert(b->is_equal(d));

    auto derived_context = d.get_context();
    static_assert(std::is_same<decltype(derived_context), cuda::any_context>::value,"");

    auto base_context = b->get_context();
    static_assert(std::is_same<decltype(base_context), cuda::any_context>::value,"");

    auto p0 = b->allocate(100);
    assert(d.events().size() == 1);
    assert((d.events().back() == event{event::ALLOCATE,
                                       reinterpret_cast<std::uintptr_t>(p0),
                                       100, derived::default_alignment}));

    auto p1 = b->allocate(42, 32);
    assert(d.events().size() == 2);
    assert(
        (d.events().back() ==
         event{event::ALLOCATE, reinterpret_cast<std::uintptr_t>(p1), 42, 32}));

    b->deallocate(p0, 100);
    assert(d.events().size() == 3);
    assert((d.events().back() == event{event::DEALLOCATE,
                                       reinterpret_cast<std::uintptr_t>(p0),
                                       100, derived::default_alignment}));

    b->deallocate(p1, 42, 32);
    assert(d.events().size() == 4);
    assert((d.events().back() == event{event::DEALLOCATE,
                                       reinterpret_cast<std::uintptr_t>(p1), 42,
                                       32}));
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
