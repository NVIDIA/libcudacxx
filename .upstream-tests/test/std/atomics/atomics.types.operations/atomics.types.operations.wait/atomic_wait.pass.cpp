//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: pre-sm-70

// <cuda/std/atomic>

#include <cuda/std/atomic>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../atomics.types.operations.req/atomic_helpers.h"
#include "concurrent_agents.h"
#include "cuda_space_selector.h"

template <class T, template<typename, typename> typename Selector, cuda::thread_scope Scope>
struct TestFn {
  __host__ __device__
  void operator()() const {
    typedef cuda::std::atomic<T> A;

#ifdef __CUDA_ARCH__
    __shared__
#endif
    A * t;
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0) {
#endif
    t = (A *)malloc(sizeof(A));
    cuda::std::atomic_init(t, T(1));
    assert(cuda::std::atomic_load(t) == T(1));
    cuda::std::atomic_wait(t, T(0));
#ifdef __CUDA_ARCH__
    }
    __syncthreads();
#endif

    auto agent_notify = LAMBDA (){
      cuda::std::atomic_store(t, T(3));
      cuda::std::atomic_notify_one(t);
    };

    auto agent_wait = LAMBDA (){
      cuda::std::atomic_wait(t, T(1));
    };

    concurrent_agents_launch(agent_notify, agent_wait);

#ifdef __CUDA_ARCH__
    __shared__
#endif
    volatile A * vt;
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0) {
#endif
    vt = (volatile A *)malloc(sizeof(A));
    cuda::std::atomic_init(vt, T(2));
    assert(cuda::std::atomic_load(vt) == T(2));
    cuda::std::atomic_wait(vt, T(1));
#ifdef __CUDA_ARCH__
    }
    __syncthreads();
#endif

    auto agent_notify_v = LAMBDA (){
      cuda::std::atomic_store(vt, T(4));
      cuda::std::atomic_notify_one(vt);
    };
    auto agent_wait_v = LAMBDA (){
      cuda::std::atomic_wait(vt, T(2));
    };

    concurrent_agents_launch(agent_notify_v, agent_wait_v);
  }
};

int main(int, char**)
{
#ifndef __CUDA_ARCH__
    cuda_thread_count = 2;
#endif

    TestEachAtomicType<TestFn, shared_memory_selector>()();

  return 0;
}
