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
// UNSUPPORTED: pre-sm-60

// <cuda/std/atomic>

#include <cuda/std/atomic>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>
#include <cuda/std/thread>

#include "test_macros.h"
#include "../atomics.types.operations.req/atomic_helpers.h"

template <class T>
struct TestFn {
  void operator()() const {
    typedef cuda::std::atomic<T> A;

    A t;
    cuda::std::atomic_init(&t, T(1));
    assert(cuda::std::atomic_load(&t) == T(1));
    cuda::std::atomic_wait(&t, T(0));
    cuda::std::thread t_([&](){
      cuda::std::atomic_store(&t, T(3));
      cuda::std::atomic_notify_one(&t);
    });
    cuda::std::atomic_wait(&t, T(1));
    t_.join();

    volatile A vt;
    cuda::std::atomic_init(&vt, T(2));
    assert(cuda::std::atomic_load(&vt) == T(2));
    cuda::std::atomic_wait(&vt, T(1));
    cuda::std::thread t2_([&](){
      cuda::std::atomic_store(&vt, T(4));
      cuda::std::atomic_notify_one(&vt);
    });
    cuda::std::atomic_wait(&vt, T(2));
    t2_.join();
  }
};

int main(int, char**)
{
    TestEachAtomicType<TestFn>()();

  return 0;
}
