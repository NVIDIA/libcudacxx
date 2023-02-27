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
// UNSUPPORTED: pre-sm-70

// <cuda/std/atomic>

#include <cuda/std/atomic>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../atomics.types.operations.req/atomic_helpers.h"
#include "concurrent_agents.h"
#include "cuda_space_selector.h"

template <class T, template<typename, typename> typename Selector, cuda::thread_scope Scope>
struct TestFn {
    __host__ __device__
    void operator()() const {
        typedef cuda::std::atomic_ref<T> A;

        {
            T * t;
            t = (T *)malloc(sizeof(T));
            A a(*t);
            a.store((T)1);
            assert(a.load() == T(1));
            a.wait((T)0);
            a.notify_one();
            a.notify_all();
        }
    }
};

int main(int, char**)
{
    TestEachAtomicRefType<TestFn>()();

    return 0;
}
