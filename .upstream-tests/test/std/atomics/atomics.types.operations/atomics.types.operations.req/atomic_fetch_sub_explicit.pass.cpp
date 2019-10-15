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

// <cuda/std/atomic>

// template <class Integral>
//     Integral
//     atomic_fetch_sub_explicit(volatile atomic<Integral>* obj, Integral op,
//                               memory_order m);
// template <class Integral>
//     Integral
//     atomic_fetch_sub_explicit(atomic<Integral>* obj, Integral op,
//                               memory_order m);
//
// template <class T>
//     T*
//     atomic_fetch_sub_explicit(volatile atomic<T*>* obj, ptrdiff_t op,
//                               memory_order m);
// template <class T>
//     T*
//     atomic_fetch_sub_explicit(atomic<T*>* obj, ptrdiff_t op, memory_order m);

#include <cuda/std/atomic>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "atomic_helpers.h"

template <class T, cuda::thread_scope>
struct TestFn {
  __host__ __device__
  void operator()() const {
    {
        typedef cuda::std::atomic<T> A;
        A t;
        cuda::std::atomic_init(&t, T(3));
        assert(cuda::std::atomic_fetch_sub_explicit(&t, T(2),
                                            cuda::std::memory_order_seq_cst) == T(3));
        assert(t == T(1));
    }
    {
        typedef cuda::std::atomic<T> A;
        volatile A t;
        cuda::std::atomic_init(&t, T(3));
        assert(cuda::std::atomic_fetch_sub_explicit(&t, T(2),
                                            cuda::std::memory_order_seq_cst) == T(3));
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
        assert(cuda::std::atomic_fetch_sub_explicit(&t, 2,
                                  cuda::std::memory_order_seq_cst) == T(3*sizeof(X)));
        assert(t == T(1*sizeof(X)));
    }
    {
        typedef cuda::std::atomic<T> A;
        typedef typename cuda::std::remove_pointer<T>::type X;
        volatile A t;
        cuda::std::atomic_init(&t, T(3*sizeof(X)));
        assert(cuda::std::atomic_fetch_sub_explicit(&t, 2,
                                  cuda::std::memory_order_seq_cst) == T(3*sizeof(X)));
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
