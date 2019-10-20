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

#ifdef __CUDA_ARCH__
  auto * tok = threadIdx.x == 1 ? new auto(b->arrive()) : nullptr;
#else
  auto * tok = new auto(b->arrive());
#endif
  auto arriver = [=] __host__ __device__ (){
    (void)b->arrive();
  };
  auto awaiter = [=] __host__ __device__ (){
    b->wait(cuda::std::move(*tok));
  };

  concurrent_agents_launch(arriver, awaiter);

#ifdef __CUDA_ARCH__
  if (threadIdx.x == 0) {
#endif
  auto tok2 = b->arrive(2);
  b->wait(cuda::std::move(tok2));
#ifdef __CUDA_ARCH__
  }
  __syncthreads();
#endif

  return 0;
}
