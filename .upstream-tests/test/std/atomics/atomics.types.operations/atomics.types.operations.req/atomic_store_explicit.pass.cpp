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

// template <class T>
//     void
//     atomic_store_explicit(volatile atomic<T>* obj, T desr, memory_order m);
//
// template <class T>
//     void
//     atomic_store_explicit(atomic<T>* obj, T desr, memory_order m);

#include <cuda/std/atomic>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "atomic_helpers.h"

template <class T, cuda::thread_scope>
struct TestFn {
  __host__ __device__
  void operator()() const {
    typedef cuda::std::atomic<T> A;
    A t;
    cuda::std::atomic_store_explicit(&t, T(1), cuda::std::memory_order_seq_cst);
    assert(t == T(1));
    volatile A vt;
    cuda::std::atomic_store_explicit(&vt, T(2), cuda::std::memory_order_seq_cst);
    assert(vt == T(2));
  }
};


int main(int, char**)
{
    TestEachAtomicType<TestFn>()();

  return 0;
}
