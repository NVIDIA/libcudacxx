//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

#include <cassert>
#include <cuda/memory_resource>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/stream_view>
#include <memory>
#include <iostream>
#include <vector>


#if __has_include(<memory_resource>)
#include <memory_resource>
namespace pmr = ::std::pmr;
#elif __has_include(<experimental/memory_resource>)
#include <experimental/memory_resource>
namespace pmr = ::std::experimental::pmr;
#endif 


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

class derived_resource : public cuda::memory_resource<cuda::memory_kind::host> {
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


template <typename Pointer>
void test_adaptor(Pointer mr){
  auto p = &*mr;
  cuda::pmr_adaptor<Pointer> adapted{std::move(mr)};
  assert(p == adapted.resource());
  assert(adapted.is_equal(adapted));

  pmr::memory_resource * std_mr = &adapted;
  assert(std_mr->is_equal(adapted));
  assert(adapted.is_equal(*std_mr));

  auto p0 = std_mr->allocate(42);
  assert(p->events().size() == 1);
  assert((p->events().back() == event{event::ALLOCATE,
                                      reinterpret_cast<std::uintptr_t>(p0), 42,
                                      alignof(std::max_align_t)}));

  std_mr->deallocate(p0, 42);
  assert(p->events().size() == 2);
  assert((p->events().back() == event{event::DEALLOCATE,
                                      reinterpret_cast<std::uintptr_t>(p0), 42,
                                      alignof(std::max_align_t)}));

  auto p1 = std_mr->allocate(42, 16);
  assert(p->events().size() == 3);
  assert(
      (p->events().back() ==
       event{event::ALLOCATE, reinterpret_cast<std::uintptr_t>(p1), 42, 16}));

  std_mr->deallocate(p1, 42, 16);
  assert(p->events().size() == 4);
  assert(
      (p->events().back() ==
       event{event::DEALLOCATE, reinterpret_cast<std::uintptr_t>(p1), 42, 16}));
}

int main(int argc, char **argv) {

#ifndef __CUDA_ARCH__
#if defined(_LIBCUDACXX_STD_PMR_NS)
   derived_resource mr_raw;
   test_adaptor(&mr_raw);
   test_adaptor(std::make_unique<derived_resource>());
   test_adaptor(std::make_shared<derived_resource>());
#endif
#endif

  return 0;
}
