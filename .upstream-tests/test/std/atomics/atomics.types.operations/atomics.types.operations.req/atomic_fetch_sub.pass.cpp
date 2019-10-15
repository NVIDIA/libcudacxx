//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
//  ... test crashes clang

// <atomic>

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

#include <atomic>
#include <type_traits>
#include <cassert>

#include "atomic_helpers.h"

template <class T, cuda::thread_scope>
struct TestFn {
  __host__ __device__
  void operator()() const {
    {
        typedef std::atomic<T> A;
        A t;
        std::atomic_init(&t, T(3));
        assert(std::atomic_fetch_sub(&t, T(2)) == T(3));
        assert(t == T(1));
    }
    {
        typedef std::atomic<T> A;
        volatile A t;
        std::atomic_init(&t, T(3));
        assert(std::atomic_fetch_sub(&t, T(2)) == T(3));
        assert(t == T(1));
    }
  }
};

template <class T>
__host__ __device__
void testp()
{
    {
        typedef std::atomic<T> A;
        typedef typename std::remove_pointer<T>::type X;
        A t;
        std::atomic_init(&t, T(3*sizeof(X)));
        assert(std::atomic_fetch_sub(&t, 2) == T(3*sizeof(X)));
        assert(t == T(1*sizeof(X)));
    }
    {
        typedef std::atomic<T> A;
        typedef typename std::remove_pointer<T>::type X;
        volatile A t;
        std::atomic_init(&t, T(3*sizeof(X)));
        assert(std::atomic_fetch_sub(&t, 2) == T(3*sizeof(X)));
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
