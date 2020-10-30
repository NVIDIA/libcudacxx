//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_member_pointer

#include <cuda/std/type_traits>
#include <cuda/std/cstddef>         // for cuda::std::nullptr_t
#include "test_macros.h"

template <class T>
__host__ __device__
void test_is_member_pointer()
{
    static_assert( cuda::std::is_member_pointer<T>::value, "");
    static_assert( cuda::std::is_member_pointer<const T>::value, "");
    static_assert( cuda::std::is_member_pointer<volatile T>::value, "");
    static_assert( cuda::std::is_member_pointer<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert( cuda::std::is_member_pointer_v<T>, "");
    static_assert( cuda::std::is_member_pointer_v<const T>, "");
    static_assert( cuda::std::is_member_pointer_v<volatile T>, "");
    static_assert( cuda::std::is_member_pointer_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__
void test_is_not_member_pointer()
{
    static_assert(!cuda::std::is_member_pointer<T>::value, "");
    static_assert(!cuda::std::is_member_pointer<const T>::value, "");
    static_assert(!cuda::std::is_member_pointer<volatile T>::value, "");
    static_assert(!cuda::std::is_member_pointer<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert(!cuda::std::is_member_pointer_v<T>, "");
    static_assert(!cuda::std::is_member_pointer_v<const T>, "");
    static_assert(!cuda::std::is_member_pointer_v<volatile T>, "");
    static_assert(!cuda::std::is_member_pointer_v<const volatile T>, "");
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
//  Arithmetic types (3.9.1), enumeration types, pointer types, pointer to member types (3.9.2),
//    cuda::std::nullptr_t, and cv-qualified versions of these types (3.9.3)
//    are collectively called scalar types.

    test_is_member_pointer<int Empty::*>();
    test_is_member_pointer<void (Empty::*)(int)>();

    test_is_not_member_pointer<cuda::std::nullptr_t>();
    test_is_not_member_pointer<void>();
    test_is_not_member_pointer<void *>();
    test_is_not_member_pointer<int>();
    test_is_not_member_pointer<int*>();
    test_is_not_member_pointer<const int*>();
    test_is_not_member_pointer<int&>();
    test_is_not_member_pointer<int&&>();
    test_is_not_member_pointer<double>();
    test_is_not_member_pointer<char[3]>();
    test_is_not_member_pointer<char[]>();
    test_is_not_member_pointer<Union>();
    test_is_not_member_pointer<Empty>();
    test_is_not_member_pointer<incomplete_type>();
    test_is_not_member_pointer<bit_zero>();
    test_is_not_member_pointer<NotEmpty>();
    test_is_not_member_pointer<Abstract>();
    test_is_not_member_pointer<int(int)>();
    test_is_not_member_pointer<Enum>();
    test_is_not_member_pointer<FunctionPtr>();

  return 0;
}
