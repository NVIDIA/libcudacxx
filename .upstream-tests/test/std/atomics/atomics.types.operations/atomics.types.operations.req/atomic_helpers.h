//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ATOMIC_HELPERS_H
#define ATOMIC_HELPERS_H

#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"

struct UserAtomicType
{
    int i;

    __host__ __device__
    explicit UserAtomicType(int d = 0) TEST_NOEXCEPT : i(d) {}

    __host__ __device__
    friend bool operator==(const UserAtomicType& x, const UserAtomicType& y)
    { return x.i == y.i; }
};

template < template <class TestArg, cuda::thread_scope> class TestFunctor, cuda::thread_scope Scope
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
    = cuda::thread_scope_system
#endif
>
struct TestEachIntegralType {
    __host__ __device__
    void operator()() const {
        TestFunctor<char, Scope>()();
        TestFunctor<signed char, Scope>()();
        TestFunctor<unsigned char, Scope>()();
        TestFunctor<short, Scope>()();
        TestFunctor<unsigned short, Scope>()();
        TestFunctor<int, Scope>()();
        TestFunctor<unsigned int, Scope>()();
        TestFunctor<long, Scope>()();
        TestFunctor<unsigned long, Scope>()();
        TestFunctor<long long, Scope>()();
        TestFunctor<unsigned long long, Scope>()();
        TestFunctor<wchar_t, Scope>();
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
        TestFunctor<char16_t, Scope>()();
        TestFunctor<char32_t, Scope>()();
#endif
        TestFunctor<  int8_t, Scope>()();
        TestFunctor< uint8_t, Scope>()();
        TestFunctor< int16_t, Scope>()();
        TestFunctor<uint16_t, Scope>()();
        TestFunctor< int32_t, Scope>()();
        TestFunctor<uint32_t, Scope>()();
        TestFunctor< int64_t, Scope>()();
        TestFunctor<uint64_t, Scope>()();
    }
};

template < template <class TestArg, cuda::thread_scope> class TestFunctor, cuda::thread_scope Scope
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
    = cuda::thread_scope_system
#endif
>
struct TestEachAtomicType {
    __host__ __device__
    void operator()() const {
        TestEachIntegralType<TestFunctor, Scope>()();
        TestFunctor<UserAtomicType, Scope>()();
        TestFunctor<int*, Scope>()();
        TestFunctor<const int*, Scope>()();
    }
};


#endif // ATOMIC_HELPER_H
