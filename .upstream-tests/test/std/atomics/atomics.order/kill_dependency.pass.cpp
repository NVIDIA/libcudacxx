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

// template <class T> T kill_dependency(T y);

#include <cuda/std/atomic>
#include <cuda/std/cassert>

int main(int, char**)
{
    assert(cuda::std::kill_dependency(5) == 5);
    assert(cuda::std::kill_dependency(-5.5) == -5.5);

  return 0;
}
