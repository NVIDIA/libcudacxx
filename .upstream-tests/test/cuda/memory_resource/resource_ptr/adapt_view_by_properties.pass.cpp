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
#include <vector>

namespace memory_kind = cuda::pmr::memory_kind;
namespace memory_access = cuda::pmr::memory_access;
using cuda::pmr::memory_resource;
using cuda::pmr::resource_ptr;
using cuda::pmr::property_list;

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

template <typename MemoryKind>
class dummy_resource : public cuda::pmr::memory_resource<MemoryKind> {
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

template <typename... Properties>
struct resource_adaptor : public memory_resource<memory_kind::with_properties<Properties...>> {
 public:
  resource_adaptor(resource_ptr<Properties...> upstream) : upstream(upstream) {
  }

 private:
  void *do_allocate(size_t size, size_t alignment) override {
    return upstream->allocate(size, alignment);
  }

  void do_deallocate(void *ptr, size_t size, size_t alignment) override {
    upstream->deallocate(ptr, size, alignment);
  }

  resource_ptr<Properties...> upstream;
};

template <typename Pointer, typename... Properties>
resource_adaptor<Properties...> adapt(cuda::pmr::basic_resource_ptr<Pointer, Properties...> v) {
  return resource_adaptor<Properties...>(v);
}

int main(int argc, char **argv) {
#ifndef __CUDA_ARCH__

  dummy_resource<memory_kind::managed> res;
  resource_ptr<cuda::pmr::of_kind<memory_kind::managed>> v1 = &res;
  resource_ptr<cuda::pmr::kind_to_properties<memory_kind::managed>> v2 = &res;
  resource_ptr<memory_access::device, memory_access::host> v3 = &res;
  cuda::pmr::resource_ptr<property_list<memory_access::device, memory_access::host>> v4 = &res;

  (void)adapt(v1);
  (void)adapt(v2);
  (void)adapt(v3);
  (void)adapt(v4);
  (void)adapt(view_resource(&res));

#endif
  return 0;
}
