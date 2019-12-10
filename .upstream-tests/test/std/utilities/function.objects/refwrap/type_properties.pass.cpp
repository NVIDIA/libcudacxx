//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// reference_wrapper

// Test that reference wrapper meets the requirements of CopyConstructible and
// CopyAssignable, and TriviallyCopyable (starting in C++14).

// Test fails due to use of is_trivially_* trait.
// XFAIL: gcc-4.9 && c++14

#include <cuda/std/functional>
#include <cuda/std/type_traits>
// #include <cuda/std/string>

#include "test_macros.h"

#if TEST_STD_VER >= 11
class MoveOnly
{
    __host__ __device__
    MoveOnly(const MoveOnly&);
    __host__ __device__
    MoveOnly& operator=(const MoveOnly&);

    int data_;
public:
    __host__ __device__
    MoveOnly(int data = 1) : data_(data) {}
    __host__ __device__
    MoveOnly(MoveOnly&& x)
        : data_(x.data_) {x.data_ = 0;}
    __host__ __device__
    MoveOnly& operator=(MoveOnly&& x)
        {data_ = x.data_; x.data_ = 0; return *this;}

    __host__ __device__
    int get() const {return data_;}
};
#endif


template <class T>
__host__ __device__
void test()
{
    typedef cuda::std::reference_wrapper<T> Wrap;
    static_assert(cuda::std::is_copy_constructible<Wrap>::value, "");
    static_assert(cuda::std::is_copy_assignable<Wrap>::value, "");
#if TEST_STD_VER >= 14
    static_assert(cuda::std::is_trivially_copyable<Wrap>::value, "");
#endif
}

int main(int, char**)
{
    test<int>();
    test<double>();
    // test<cuda::std::string>();
#if TEST_STD_VER >= 11
    test<MoveOnly>();
#endif

  return 0;
}
