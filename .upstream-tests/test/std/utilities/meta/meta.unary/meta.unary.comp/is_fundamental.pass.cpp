//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_fundamental

#include <cuda/std/type_traits>
#include <cuda/std/cstddef>         // for cuda::std::nullptr_t
#include "test_macros.h"

#pragma diag_suppress cuda_demote_unsupported_floating_point

template <class T>
__host__ __device__
void test_is_fundamental()
{
    static_assert( cuda::std::is_fundamental<T>::value, "");
    static_assert( cuda::std::is_fundamental<const T>::value, "");
    static_assert( cuda::std::is_fundamental<volatile T>::value, "");
    static_assert( cuda::std::is_fundamental<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert( cuda::std::is_fundamental_v<T>, "");
    static_assert( cuda::std::is_fundamental_v<const T>, "");
    static_assert( cuda::std::is_fundamental_v<volatile T>, "");
    static_assert( cuda::std::is_fundamental_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__
void test_is_not_fundamental()
{
    static_assert(!cuda::std::is_fundamental<T>::value, "");
    static_assert(!cuda::std::is_fundamental<const T>::value, "");
    static_assert(!cuda::std::is_fundamental<volatile T>::value, "");
    static_assert(!cuda::std::is_fundamental<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert(!cuda::std::is_fundamental_v<T>, "");
    static_assert(!cuda::std::is_fundamental_v<const T>, "");
    static_assert(!cuda::std::is_fundamental_v<volatile T>, "");
    static_assert(!cuda::std::is_fundamental_v<const volatile T>, "");
#endif
}

class incomplete_type;

class Empty
{
};

class NotEmpty
{
    __host__ __device__
    virtual ~NotEmpty();
};

union Union {};

struct bit_zero
{
    int :  0;
};

class Abstract
{
    __host__ __device__
    virtual ~Abstract() = 0;
};

enum Enum {zero, one};

typedef void (*FunctionPtr)();


int main(int, char**)
{
    test_is_fundamental<cuda::std::nullptr_t>();
    test_is_fundamental<void>();
    test_is_fundamental<short>();
    test_is_fundamental<unsigned short>();
    test_is_fundamental<int>();
    test_is_fundamental<unsigned int>();
    test_is_fundamental<long>();
    test_is_fundamental<unsigned long>();
    test_is_fundamental<long long>();
    test_is_fundamental<unsigned long long>();
    test_is_fundamental<bool>();
    test_is_fundamental<char>();
    test_is_fundamental<signed char>();
    test_is_fundamental<unsigned char>();
    test_is_fundamental<wchar_t>();
    test_is_fundamental<double>();
    test_is_fundamental<float>();
    test_is_fundamental<double>();
    test_is_fundamental<long double>();
    test_is_fundamental<char16_t>();
    test_is_fundamental<char32_t>();

    test_is_not_fundamental<char[3]>();
    test_is_not_fundamental<char[]>();
    test_is_not_fundamental<void *>();
    test_is_not_fundamental<FunctionPtr>();
    test_is_not_fundamental<int&>();
    test_is_not_fundamental<int&&>();
    test_is_not_fundamental<Union>();
    test_is_not_fundamental<Empty>();
    test_is_not_fundamental<incomplete_type>();
    test_is_not_fundamental<bit_zero>();
    test_is_not_fundamental<int*>();
    test_is_not_fundamental<const int*>();
    test_is_not_fundamental<Enum>();
    test_is_not_fundamental<NotEmpty>();
    test_is_not_fundamental<Abstract>();

  return 0;
}
