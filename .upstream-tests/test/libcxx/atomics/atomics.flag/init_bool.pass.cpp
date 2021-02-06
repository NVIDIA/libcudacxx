//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/std/atomic>

// struct atomic_flag

// TESTING EXTENSION atomic_flag(bool)

#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "cuda_space_selector.h"

template<template<typename, typename> class Selector>
__host__ __device__
void test()
{
    {
        Selector<cuda::std::atomic_flag, constructor_initializer> sel;
        cuda::std::atomic_flag & f = *sel.construct(false);
        assert(f.test_and_set() == 0);
    }
    {
        Selector<cuda::std::atomic_flag, constructor_initializer> sel;
        cuda::std::atomic_flag & f = *sel.construct(true);
        assert(f.test_and_set() == 1);
    }
}

int main(int, char**)
{
    _LIBCUDACXX_CUDA_DISPATCH(
        HOST, _LIBCUDACXX_ARCH_BLOCK(
            test<local_memory_selector>();
        )
        GREATER_THAN_SM62, _LIBCUDACXX_ARCH_BLOCK(
            test<local_memory_selector>();
        ),
        DEVICE, _LIBCUDACXX_ARCH_BLOCK(
            test<shared_memory_selector>();
            test<global_memory_selector>();
        )
    )

  return 0;
}
