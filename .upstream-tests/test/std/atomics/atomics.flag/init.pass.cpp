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
// XFAIL: c++98, c++03

// <cuda/std/atomic>

// struct atomic_flag

// atomic_flag() = ATOMIC_FLAG_INIT;

#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    cuda::std::atomic_flag f = ATOMIC_FLAG_INIT;
    assert(f.test_and_set() == 0);
#endif

  return 0;
}
