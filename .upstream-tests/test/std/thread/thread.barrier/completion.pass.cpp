//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: pre-sm-60

// <cuda/std/barrier>

#include <cuda/std/barrier>

#include "test_macros.h"
#include "concurrent_agents.h"

int main(int, char**)
{
#ifndef __CUDA_ARCH__
    cuda_thread_count = 2;
#endif

#ifdef __CUDA_ARCH__
  __shared__
#endif
  int * x;
#ifdef __CUDA_ARCH__
  if (threadIdx.x == 0) {
#endif
  x = (int *)malloc(sizeof(int));
  *x = 0;
#ifdef __CUDA_ARCH__
  }
  __syncthreads();
#endif

  auto comp = [=] __host__ __device__ () { *x += 1; };

#ifdef __CUDA_ARCH__
  __shared__
#endif
  cuda::std::barrier<decltype(comp)> * b;
#ifdef __CUDA_ARCH__
  if (threadIdx.x == 0) {
#endif
  b = new cuda::std::barrier<decltype(comp)>(2, comp);
#ifdef __CUDA_ARCH__
  }
  __syncthreads();
#endif

  auto worker = [=] __host__ __device__ () {
      for(int i = 0; i < 10; ++i)
        b->arrive_and_wait();
      assert(*x == 10);
  };

  concurrent_agents_launch(worker, worker);

  assert(*x == 10);
  return 0;
}
