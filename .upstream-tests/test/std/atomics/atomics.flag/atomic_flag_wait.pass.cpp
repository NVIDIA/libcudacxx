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
// NVC++ does not support GPU function pointers
// UNSUPPORTED: pgi

// <cuda/std/atomic>

#include <cuda/std/atomic>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "concurrent_agents.h"
#include "cuda_space_selector.h"

template<template<typename, typename> class Selector>
__host__ __device__
void test()
{
#ifdef __CUDA_ARCH__
    __shared__
#endif
    cuda::std::atomic_flag * t;
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0) {
#endif
    t = new cuda::std::atomic_flag();
    cuda::std::atomic_flag_clear(t);
    cuda::std::atomic_flag_wait(t, true);
#ifdef __CUDA_ARCH__
    }
    __syncthreads();
#endif

    auto agent_notify = LAMBDA (){
        assert(cuda::std::atomic_flag_test_and_set(t) == false);
        cuda::std::atomic_flag_notify_one(t);
    };

    auto agent_wait = LAMBDA (){
        cuda::std::atomic_flag_wait(t, false);
    };

    concurrent_agents_launch(agent_notify, agent_wait);

#ifdef __CUDA_ARCH__
    __shared__
#endif
    volatile cuda::std::atomic_flag * vt;
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0) {
#endif
    vt = new cuda::std::atomic_flag();
    cuda::std::atomic_flag_clear(vt);
    cuda::std::atomic_flag_wait(vt, true);
#ifdef __CUDA_ARCH__
    }
    __syncthreads();
#endif

    auto agent_notify_v = LAMBDA (){
        assert(cuda::std::atomic_flag_test_and_set(vt) == false);
        cuda::std::atomic_flag_notify_one(vt);
    };

    auto agent_wait_v = LAMBDA (){
        cuda::std::atomic_flag_wait(vt, false);
    };

    concurrent_agents_launch(agent_notify_v, agent_wait_v);
}

int main(int, char**)
{
#ifndef __CUDA_ARCH__
    cuda_thread_count = 2;
#endif

    test<shared_memory_selector>();

  return 0;
}
