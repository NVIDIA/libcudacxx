//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: c++98, c++03

// NOTE: atomic<> of a TriviallyCopyable class is wrongly rejected by older
// clang versions. It was fixed right before the llvm 3.5 release. See PR18097.
// XFAIL: apple-clang-6.0, clang-3.4, clang-3.3

// <cuda/std/atomic>

// constexpr atomic<T>::atomic(T value)

#include <cuda/std/atomic>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "atomic_helpers.h"

struct UserType {
    int i;

    __host__ __device__
    UserType() noexcept {}
    __host__ __device__
    constexpr explicit UserType(int d) noexcept : i(d) {}

    __host__ __device__
    friend bool operator==(const UserType& x, const UserType& y) {
        return x.i == y.i;
    }
};

template <class Tp, cuda::thread_scope Scope>
struct TestFunc {
    __host__ __device__
    void operator()() const {
        typedef cuda::atomic<Tp, Scope> Atomic;
        static_assert(cuda::std::is_literal_type<Atomic>::value, "");
        constexpr Tp t(42);
        {
            constexpr Atomic a(t);
            assert(a == t);
        }
        {
            constexpr Atomic a{t};
            assert(a == t);
        }
        #if !defined(_GNUC_VER) || _GNUC_VER >= 409
        // TODO: Figure out why this is failing with GCC 4.8.2 on CentOS 7 only.
        {
            constexpr Atomic a = ATOMIC_VAR_INIT(t);
            assert(a == t);
        }
        #endif
    }
};


int main(int, char**)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
    TestFunc<UserType, cuda::thread_scope_system>()();
    TestEachIntegralType<TestFunc, cuda::thread_scope_system>()();
#endif
    TestFunc<UserType, cuda::thread_scope_device>()();
    TestEachIntegralType<TestFunc, cuda::thread_scope_device>()();
    TestFunc<UserType, cuda::thread_scope_block>()();
    TestEachIntegralType<TestFunc, cuda::thread_scope_block>()();

  return 0;
}
