//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: pre-sm-70

// <cuda/std/barrier>

#include <cuda/std/barrier>

#include "test_macros.h"
#include "concurrent_agents.h"
#include "cuda_space_selector.h"

template<typename Barrier,
    template<typename, typename> typename Selector,
    typename Initializer = constructor_initializer>
__host__ __device__
void test()
{
  Selector<Barrier, Initializer> sel;
  SHARED Barrier * b;
  b = sel.construct(2);

  auto dropper = LAMBDA (){
    b->arrive_and_drop();
  };

  auto arriver = LAMBDA (){
    b->arrive_and_wait();
    b->arrive_and_wait();
  };

  concurrent_agents_launch(dropper, arriver);
}

int main(int, char**)
{
#ifndef __CUDA_ARCH__
  cuda_thread_count = 2;

  test<cuda::std::barrier<>, local_memory_selector>();
  test<cuda::barrier<cuda::thread_scope_block>, local_memory_selector>();
  test<cuda::barrier<cuda::thread_scope_device>, local_memory_selector>();
  test<cuda::barrier<cuda::thread_scope_system>, local_memory_selector>();
#else
  test<cuda::std::barrier<>, shared_memory_selector>();
  test<cuda::barrier<cuda::thread_scope_block>, shared_memory_selector>();
  test<cuda::barrier<cuda::thread_scope_device>, shared_memory_selector>();
  test<cuda::barrier<cuda::thread_scope_system>, shared_memory_selector>();

  test<cuda::std::barrier<>, global_memory_selector>();
  test<cuda::barrier<cuda::thread_scope_block>, global_memory_selector>();
  test<cuda::barrier<cuda::thread_scope_device>, global_memory_selector>();
  test<cuda::barrier<cuda::thread_scope_system>, global_memory_selector>();
#endif

  return 0;
}
