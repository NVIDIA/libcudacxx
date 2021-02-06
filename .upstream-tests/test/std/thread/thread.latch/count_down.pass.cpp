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

// <cuda/std/latch>

#include <cuda/std/latch>

#include "test_macros.h"
#include "concurrent_agents.h"
#include "cuda_space_selector.h"

template<typename Latch,
    template<typename, typename> typename Selector,
    typename Initializer = constructor_initializer>
__host__ __device__
void test()
{
  Selector<Latch, Initializer> sel;
  SHARED Latch * l;
  l = sel.construct(2);

  _LIBCUDACXX_CUDA_DISPATCH(
      DEVICE, _LIBCUDACXX_ARCH_BLOCK(
          if (threadIdx.x == 0) {
              l->count_down();
          }
          __syncthreads();
      ),
      HOST, _LIBCUDACXX_ARCH_BLOCK(
              l->count_down();
      )
  )

  auto count_downer = LAMBDA (){
    l->count_down();
  };

  auto awaiter = LAMBDA (){
    l->wait();
  };

  concurrent_agents_launch(awaiter, count_downer);
}

int main(int, char**)
{
    _LIBCUDACXX_CUDA_DISPATCH(
      HOST, _LIBCUDACXX_ARCH_BLOCK(
        cuda_thread_count = 2;

        test<cuda::std::latch, local_memory_selector>();
        test<cuda::latch<cuda::thread_scope_block>, local_memory_selector>();
        test<cuda::latch<cuda::thread_scope_device>, local_memory_selector>();
        test<cuda::latch<cuda::thread_scope_system>, local_memory_selector>();
      ),
      DEVICE, _LIBCUDACXX_ARCH_BLOCK(
        test<cuda::std::latch, shared_memory_selector>();
        test<cuda::latch<cuda::thread_scope_block>, shared_memory_selector>();
        test<cuda::latch<cuda::thread_scope_device>, shared_memory_selector>();
        test<cuda::latch<cuda::thread_scope_system>, shared_memory_selector>();

        test<cuda::std::latch, global_memory_selector>();
        test<cuda::latch<cuda::thread_scope_block>, global_memory_selector>();
        test<cuda::latch<cuda::thread_scope_device>, global_memory_selector>();
        test<cuda::latch<cuda::thread_scope_system>, global_memory_selector>();
      )
    )

  return 0;
}
