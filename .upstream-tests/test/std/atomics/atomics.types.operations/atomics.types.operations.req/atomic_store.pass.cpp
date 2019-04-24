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

// template <class T>
//     void
//     atomic_store(volatile atomic<T>* obj, T desr);
//
// template <class T>
//     void
//     atomic_store(atomic<T>* obj, T desr);

#include <cuda/std/atomic>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "atomic_helpers.h"

template <class T>
struct TestFn {
  __host__ __device__
  void operator()() const {
    typedef cuda::std::atomic<T> A;
    A t;
    cuda::std::atomic_store(&t, T(1));
    assert(t == T(1));
    volatile A vt;
    cuda::std::atomic_store(&vt, T(2));
    assert(vt == T(2));
  }
};


int main(int, char**)
{
    TestEachAtomicType<TestFn>()();

  return 0;
}
