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

int main(int, char**)
{
  cuda::std::barrier b(2);

  auto tok = b.arrive();
  cuda::std::thread t([&](){
    (void)b.arrive();
  });
  b.wait(cuda::std::move(tok));
  t.join();

  auto tok2 = b.arrive(2);
  b.wait(cuda::std::move(tok2));
  return 0;
}
