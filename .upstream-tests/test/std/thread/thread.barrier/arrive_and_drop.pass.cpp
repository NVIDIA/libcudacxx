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
  cuda::std::barrier<> * b;
#ifdef __CUDA_ARCH__
  if (threadIdx.x == 0) {
#endif
  b = new cuda::std::barrier<>(2);
#ifdef __CUDA_ARCH__
  }
  __syncthreads();
#endif

  auto dropper = [=] __host__ __device__ (){
    b->arrive_and_drop();
  };

  auto arriver = [=] __host__ __device__ (){
    b->arrive_and_wait();
    b->arrive_and_wait();
  };

  concurrent_agents_launch(dropper, arriver);

  return 0;
}
