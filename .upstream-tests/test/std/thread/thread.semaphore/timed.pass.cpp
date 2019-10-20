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
#include <cuda/std/chrono>
#include <thread>

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
  s = new cuda::std::counting_semaphore<>(0);
#ifdef __CUDA_ARCH__
  }
  __syncthreads();
#endif

  auto const start = cuda::std::chrono::steady_clock::now();

#ifdef __CUDA_ARCH__
  if (threadIdx.x == 0) {
#endif
  assert(!s->try_acquire_until(start + cuda::std::chrono::milliseconds(250)));
  assert(!s->try_acquire_for(cuda::std::chrono::milliseconds(250)));
#ifdef __CUDA_ARCH__
  }
  __syncthreads();
#endif

  auto releaser = [=] __host__ __device__ (){
    //cuda::std::this_thread::sleep_for(cuda::std::chrono::milliseconds(250));
    s->release();
    //cuda::std::this_thread::sleep_for(cuda::std::chrono::milliseconds(250));
    s->release();
  };

  auto acquirer = [=] __host__ __device__ (){
    assert(s->try_acquire_until(start + cuda::std::chrono::seconds(2)));
    assert(s->try_acquire_for(cuda::std::chrono::seconds(2)));
  };

  concurrent_agents_launch(acquirer, releaser);

#ifdef __CUDA_ARCH__
  if (threadIdx.x == 0) {
#endif
  auto const end = cuda::std::chrono::steady_clock::now();
  assert(end - start < cuda::std::chrono::seconds(10));
#ifdef __CUDA_ARCH__
  }
#endif

  return 0;
}
