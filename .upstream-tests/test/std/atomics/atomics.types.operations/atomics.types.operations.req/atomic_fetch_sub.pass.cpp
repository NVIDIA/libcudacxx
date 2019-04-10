//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
//  ... test crashes clang

// <cuda/std/atomic>

// template <class Integral>
//     Integral
//     atomic_fetch_sub(volatile atomic<Integral>* obj, Integral op);
//
// template <class Integral>
//     Integral
//     atomic_fetch_sub(atomic<Integral>* obj, Integral op);
//
// template <class T>
//     T*
//     atomic_fetch_sub(volatile atomic<T*>* obj, ptrdiff_t op);
//
// template <class T>
//     T*
//     atomic_fetch_sub(atomic<T*>* obj, ptrdiff_t op);

#include <cuda/std/atomic>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "atomic_helpers.h"

template <class T>
struct TestFn {
  __host__ __device__
  void operator()() const {
    {
        typedef cuda::std::atomic<T> A;
        A t;
        cuda::std::atomic_init(&t, T(3));
        assert(cuda::std::atomic_fetch_sub(&t, T(2)) == T(3));
        assert(t == T(1));
    }
    {
        typedef cuda::std::atomic<T> A;
        volatile A t;
        cuda::std::atomic_init(&t, T(3));
        assert(cuda::std::atomic_fetch_sub(&t, T(2)) == T(3));
        assert(t == T(1));
    }
  }
};

template <class T>
__host__ __device__
void testp()
{
    {
        typedef cuda::std::atomic<T> A;
        typedef typename cuda::std::remove_pointer<T>::type X;
        A t;
        cuda::std::atomic_init(&t, T(3*sizeof(X)));
        assert(cuda::std::atomic_fetch_sub(&t, 2) == T(3*sizeof(X)));
        assert(t == T(1*sizeof(X)));
    }
    {
        typedef cuda::std::atomic<T> A;
        typedef typename cuda::std::remove_pointer<T>::type X;
        volatile A t;
        cuda::std::atomic_init(&t, T(3*sizeof(X)));
        assert(cuda::std::atomic_fetch_sub(&t, 2) == T(3*sizeof(X)));
        assert(t == T(1*sizeof(X)));
    }
}

int main(int, char**)
{
    TestEachIntegralType<TestFn>()();
    testp<int*>();
    testp<const int*>();

  return 0;
}
