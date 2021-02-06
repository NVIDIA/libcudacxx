//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70
//  ... assertion fails line 34

// <cuda/std/atomic>

// template <class T>
//     bool
//     atomic_compare_exchange_strong(volatile atomic<T>* obj, T* expc, T desr);
//
// template <class T>
//     bool
//     atomic_compare_exchange_strong(atomic<T>* obj, T* expc, T desr);

#include <cuda/std/atomic>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "atomic_helpers.h"
#include "cuda_space_selector.h"

template <class T, template<typename, typename> typename Selector, cuda::thread_scope>
struct TestFn {
  __host__ __device__
  void operator()() const {
    {
        typedef cuda::std::atomic<T> A;
        Selector<A, constructor_initializer> sel;
        A & a = *sel.construct();
        T t(T(1));
        cuda::std::atomic_init(&a, t);
        assert(cuda::std::atomic_compare_exchange_strong(&a, &t, T(2)) == true);
        assert(a == T(2));
        assert(t == T(1));
        assert(cuda::std::atomic_compare_exchange_strong(&a, &t, T(3)) == false);
        assert(a == T(2));
        assert(t == T(2));
    }
    {
        typedef cuda::std::atomic<T> A;
        Selector<volatile A, constructor_initializer> sel;
        volatile A & a = *sel.construct();
        T t(T(1));
        cuda::std::atomic_init(&a, t);
        assert(cuda::std::atomic_compare_exchange_strong(&a, &t, T(2)) == true);
        assert(a == T(2));
        assert(t == T(1));
        assert(cuda::std::atomic_compare_exchange_strong(&a, &t, T(3)) == false);
        assert(a == T(2));
        assert(t == T(2));
    }
  }
};

int main(int, char**)
{
    _LIBCUDACXX_CUDA_DISPATCH(
        GREATER_THAN_SM62, _LIBCUDACXX_ARCH_BLOCK(
            TestEachAtomicType<TestFn, local_memory_selector>()();
        ),
        HOST, _LIBCUDACXX_ARCH_BLOCK(
            TestEachAtomicType<TestFn, local_memory_selector>()();
        ),
        DEVICE, _LIBCUDACXX_ARCH_BLOCK(
            TestEachAtomicType<TestFn, shared_memory_selector>()();
            TestEachAtomicType<TestFn, global_memory_selector>()();
        )
    )

  return 0;
}
