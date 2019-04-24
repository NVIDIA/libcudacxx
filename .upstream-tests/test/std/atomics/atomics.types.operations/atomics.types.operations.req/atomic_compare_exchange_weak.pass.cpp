//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-70
//  ... assertion fails line 34

// <cuda/std/atomic>

// template <class T>
//     bool
//     atomic_compare_exchange_weak(volatile atomic<T>* obj, T* expc, T desr);
//
// template <class T>
//     bool
//     atomic_compare_exchange_weak(atomic<T>* obj, T* expc, T desr);

#include <cuda/std/atomic>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include <cmpxchg_loop.h>
#include "atomic_helpers.h"

template <class T>
struct TestFn {
  __host__ __device__
  void operator()() const {
    {
        typedef cuda::std::atomic<T> A;
        A a;
        T t(T(1));
        cuda::std::atomic_init(&a, t);
        assert(c_cmpxchg_weak_loop(&a, &t, T(2)) == true);
        assert(a == T(2));
        assert(t == T(1));
        assert(cuda::std::atomic_compare_exchange_weak(&a, &t, T(3)) == false);
        assert(a == T(2));
        assert(t == T(2));
    }
    {
        typedef cuda::std::atomic<T> A;
        volatile A a;
        T t(T(1));
        cuda::std::atomic_init(&a, t);
        assert(c_cmpxchg_weak_loop(&a, &t, T(2)) == true);
        assert(a == T(2));
        assert(t == T(1));
        assert(cuda::std::atomic_compare_exchange_weak(&a, &t, T(3)) == false);
        assert(a == T(2));
        assert(t == T(2));
    }
  }
};

int main(int, char**)
{
    TestEachAtomicType<TestFn>()();

  return 0;
}
