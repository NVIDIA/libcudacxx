//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60

// <cuda/std/atomic>

// struct atomic_flag

// bool atomic_flag_test_and_set(volatile atomic_flag*);
// bool atomic_flag_test_and_set(atomic_flag*);

#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        cuda::std::atomic_flag f;
        f.clear();
        assert(atomic_flag_test_and_set(&f) == 0);
        assert(f.test_and_set() == 1);
    }
    {
        volatile cuda::std::atomic_flag f;
        f.clear();
        assert(atomic_flag_test_and_set(&f) == 0);
        assert(f.test_and_set() == 1);
    }

  return 0;
}
