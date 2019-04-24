//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-70
//  ... assertion fails line 35

// <cuda/std/atomic>

// template <class T>
//     T
//     atomic_load(const volatile atomic<T>* obj);
//
// template <class T>
//     T
//     atomic_load(const atomic<T>* obj);

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
    cuda::std::atomic_init(&t, T(1));
    assert(cuda::std::atomic_load(&t) == T(1));
    volatile A vt;
    cuda::std::atomic_init(&vt, T(2));
    assert(cuda::std::atomic_load(&vt) == T(2));
  }
};

int main(int, char**)
{
    TestEachAtomicType<TestFn>()();

  return 0;
}
