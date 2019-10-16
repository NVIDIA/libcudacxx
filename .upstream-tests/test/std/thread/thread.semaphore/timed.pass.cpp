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
#include <cuda/std/chrono>

#include "test_macros.h"

int main(int, char**)
{
  auto const start = cuda::std::chrono::steady_clock::now();

  cuda::std::counting_semaphore s(0);

  assert(!s.try_acquire_until(start + cuda::std::chrono::milliseconds(250)));
  assert(!s.try_acquire_for(cuda::std::chrono::milliseconds(250)));

  cuda::std::thread t([&](){
    cuda::std::this_thread::sleep_for(cuda::std::chrono::milliseconds(250));
    s.release();
    cuda::std::this_thread::sleep_for(cuda::std::chrono::milliseconds(250));
    s.release();
  });

  assert(s.try_acquire_until(start + cuda::std::chrono::seconds(2)));
  assert(s.try_acquire_for(cuda::std::chrono::seconds(2)));
  t.join();

  auto const end = cuda::std::chrono::steady_clock::now();
  assert(end - start < cuda::std::chrono::seconds(10));

  return 0;
}
