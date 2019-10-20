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

// <cuda/std/semaphore>

#include <cuda/std/semaphore>

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
  cuda::std::counting_semaphore<> * s;
#ifdef __CUDA_ARCH__
  if (threadIdx.x == 0) {
#endif
  s = new cuda::std::counting_semaphore<>(2);
#ifdef __CUDA_ARCH__
  }
  __syncthreads();
#endif

#ifdef __CUDA_ARCH__
  if (threadIdx.x == 1) {
#else
  std::thread t([&](){
#endif
    s->acquire();
#ifdef __CUDA_ARCH__
  }
  __syncthreads();
#else
  });
  t.join();
#endif

#ifdef __CUDA_ARCH__
  if (threadIdx.x == 0) {
#endif
  s->acquire();
#ifdef __CUDA_ARCH__
  }
#endif

  return 0;
}
