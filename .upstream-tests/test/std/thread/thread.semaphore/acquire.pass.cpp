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
#include <cuda/std/thread>

#include "test_macros.h"

int main(int, char**)
{
  cuda::std::counting_semaphore s(2);

  cuda::std::thread t([&](){
    s.acquire();
  });
  t.join();

  s.acquire();

  return 0;
}
