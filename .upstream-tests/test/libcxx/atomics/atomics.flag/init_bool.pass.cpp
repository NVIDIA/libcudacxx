//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-70

// <cuda/std/atomic>

// struct atomic_flag

// TESTING EXTENSION atomic_flag(bool)

#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        cuda::std::atomic_flag f(false);
        assert(f.test_and_set() == 0);
    }
    {
        cuda::std::atomic_flag f(true);
        assert(f.test_and_set() == 1);
    }

  return 0;
}
