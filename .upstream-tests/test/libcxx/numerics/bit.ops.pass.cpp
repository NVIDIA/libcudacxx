//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Test the __XXXX routines in the <cuda/std/bit> header.
// These are not supposed to be exhaustive tests, just sanity checks.

#include <cuda/std/bit>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char **)
{

    
#if TEST_STD_VER > 11
    {
    constexpr unsigned v = 0x1237U;

//  These are all constexpr in C++14 and later
    static_assert( cuda::std::__rotl(v, 4) == 0x00012370, "");
    static_assert( cuda::std::__rotr(v, 4) == 0x70000123, "");
    static_assert( cuda::std::__countl_one(v)  == 0, "");
    static_assert( cuda::std::__countr_one(v)  == 3, "");
    static_assert( cuda::std::__countl_zero(v) == 19, "");
    static_assert( cuda::std::__countr_zero(v) == 0, "");
    
    static_assert( cuda::std::__libcpp_popcount(v) == 7, "");
    static_assert( cuda::std::__bit_log2(v) == 12, "");
    static_assert(!cuda::std::__ispow2(v), "");
    }
#endif

    {
    const unsigned v = 0x12345678;

    ASSERT_SAME_TYPE(unsigned, decltype(cuda::std::__rotl(v, 3)));
    ASSERT_SAME_TYPE(unsigned, decltype(cuda::std::__rotr(v, 3)));

    ASSERT_SAME_TYPE(int, decltype(cuda::std::__countl_one(v)));
    ASSERT_SAME_TYPE(int, decltype(cuda::std::__countr_one(v)));
    ASSERT_SAME_TYPE(int, decltype(cuda::std::__countl_zero(v)));
    ASSERT_SAME_TYPE(int, decltype(cuda::std::__countr_zero(v)));

    ASSERT_SAME_TYPE(int,      decltype(cuda::std::__libcpp_popcount(v)));
    ASSERT_SAME_TYPE(unsigned, decltype(cuda::std::__bit_log2(v)));
    ASSERT_SAME_TYPE(bool,     decltype(cuda::std::__ispow2(v)));


    assert( cuda::std::__rotl(v, 3) == 0x91a2b3c0U);
    assert( cuda::std::__rotr(v, 3) == 0x02468acfU);

    assert( cuda::std::__countl_one(v)  == 0);
    assert( cuda::std::__countr_one(v)  == 0);
    assert( cuda::std::__countl_zero(v) == 3);
    assert( cuda::std::__countr_zero(v) == 3);

    assert( cuda::std::__libcpp_popcount(v) == 13);
    assert( cuda::std::__bit_log2(v) == 28);
    assert(!cuda::std::__ispow2(v));  
    }
    
    return 0;
}

