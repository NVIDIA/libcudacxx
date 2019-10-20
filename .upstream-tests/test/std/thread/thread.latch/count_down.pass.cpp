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

// <cuda/std/latch>

#include <cuda/std/latch>

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
  cuda::std::latch * l;
#ifdef __CUDA_ARCH__
  if (threadIdx.x == 0) {
#endif
  l = new cuda::std::latch(2);
#ifdef __CUDA_ARCH__
  }
  __syncthreads();
#endif

#ifdef __CUDA_ARCH__
  if (threadIdx.x == 0) {
#endif
  l->count_down();
#ifdef __CUDA_ARCH__
  }
  __syncthreads();
#endif
  auto count_downer = [=] __host__ __device__ (){
    l->count_down();
  };

  auto awaiter = [=] __host__ __device__ (){
    l->wait();
  };

  concurrent_agents_launch(awaiter, count_downer);

  return 0;
}
