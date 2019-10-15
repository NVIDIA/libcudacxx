//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17 

// template <class T>
//   constexpr int countr_one(T x) noexcept;

// Returns: The number of consecutive 1 bits, starting from the least significant bit.
//   [ Note: Returns N if x == cuda::std::numeric_limits<T>::max(). ]
//
// Remarks: This function shall not participate in overload resolution unless 
//	T is an unsigned integer type

#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

class A{};
enum       E1 : unsigned char { rEd };
enum class E2 : unsigned char { red };

template <typename T>
constexpr bool constexpr_test()
{
	const int dig = cuda::std::numeric_limits<T>::digits;
	return cuda::std::countr_one(T(0)) == 0
	   &&  cuda::std::countr_one(T(1)) == 1
	   &&  cuda::std::countr_one(T(2)) == 0
	   &&  cuda::std::countr_one(T(3)) == 2
	   &&  cuda::std::countr_one(T(4)) == 0
	   &&  cuda::std::countr_one(T(5)) == 1
	   &&  cuda::std::countr_one(T(6)) == 0
	   &&  cuda::std::countr_one(T(7)) == 3
	   &&  cuda::std::countr_one(T(8)) == 0
	   &&  cuda::std::countr_one(T(9)) == 1
	   &&  cuda::std::countr_one(cuda::std::numeric_limits<T>::max()) == dig
	  ;
}


template <typename T>
void runtime_test()
{
	ASSERT_SAME_TYPE(int, decltype(cuda::std::countr_one(T(0))));
	ASSERT_NOEXCEPT(               cuda::std::countr_one(T(0)));
	
	assert( cuda::std::countr_one(T(121)) == 1);
	assert( cuda::std::countr_one(T(122)) == 0);
	assert( cuda::std::countr_one(T(123)) == 2);
	assert( cuda::std::countr_one(T(124)) == 0);
	assert( cuda::std::countr_one(T(125)) == 1);
	assert( cuda::std::countr_one(T(126)) == 0);
	assert( cuda::std::countr_one(T(127)) == 7);
	assert( cuda::std::countr_one(T(128)) == 0);
	assert( cuda::std::countr_one(T(129)) == 1);
	assert( cuda::std::countr_one(T(130)) == 0);
}

int main()
{
	
    {
    auto lambda = [](auto x) -> decltype(cuda::std::countr_one(x)) {};
    using L = decltype(lambda);
    
    static_assert( cuda::std::is_invocable_v<L, unsigned char>, "");
    static_assert( cuda::std::is_invocable_v<L, unsigned int>, "");
    static_assert( cuda::std::is_invocable_v<L, unsigned long>, "");
    static_assert( cuda::std::is_invocable_v<L, unsigned long long>, "");

    static_assert( cuda::std::is_invocable_v<L, uint8_t>, "");
    static_assert( cuda::std::is_invocable_v<L, uint16_t>, "");
    static_assert( cuda::std::is_invocable_v<L, uint32_t>, "");
    static_assert( cuda::std::is_invocable_v<L, uint64_t>, "");
    static_assert( cuda::std::is_invocable_v<L, size_t>, "");

    static_assert( cuda::std::is_invocable_v<L, uintmax_t>, "");
    static_assert( cuda::std::is_invocable_v<L, uintptr_t>, "");


    static_assert(!cuda::std::is_invocable_v<L, int>, "");
    static_assert(!cuda::std::is_invocable_v<L, signed int>, "");
    static_assert(!cuda::std::is_invocable_v<L, long>, "");
    static_assert(!cuda::std::is_invocable_v<L, long long>, "");

    static_assert(!cuda::std::is_invocable_v<L, int8_t>, "");
    static_assert(!cuda::std::is_invocable_v<L, int16_t>, "");
    static_assert(!cuda::std::is_invocable_v<L, int32_t>, "");
    static_assert(!cuda::std::is_invocable_v<L, int64_t>, "");
    static_assert(!cuda::std::is_invocable_v<L, ptrdiff_t>, "");

    static_assert(!cuda::std::is_invocable_v<L, bool>, "");
    static_assert(!cuda::std::is_invocable_v<L, signed char>, "");
    static_assert(!cuda::std::is_invocable_v<L, char16_t>, "");
    static_assert(!cuda::std::is_invocable_v<L, char32_t>, "");

#ifndef _LIBCUDACXX_HAS_NO_INT128
    static_assert( cuda::std::is_invocable_v<L, __uint128_t>, "");
    static_assert(!cuda::std::is_invocable_v<L, __int128_t>, "");
#endif
 
    static_assert(!cuda::std::is_invocable_v<L, A>, "");
    static_assert(!cuda::std::is_invocable_v<L, E1>, "");
    static_assert(!cuda::std::is_invocable_v<L, E2>, "");
    }

	static_assert(constexpr_test<unsigned char>(),      "");
	static_assert(constexpr_test<unsigned short>(),     "");
	static_assert(constexpr_test<unsigned>(),           "");
	static_assert(constexpr_test<unsigned long>(),      "");
	static_assert(constexpr_test<unsigned long long>(), "");

	static_assert(constexpr_test<uint8_t>(),   "");
	static_assert(constexpr_test<uint16_t>(),  "");
	static_assert(constexpr_test<uint32_t>(),  "");
	static_assert(constexpr_test<uint64_t>(),  "");
	static_assert(constexpr_test<size_t>(),    "");
	static_assert(constexpr_test<uintmax_t>(), "");
	static_assert(constexpr_test<uintptr_t>(), "");

#ifndef _LIBCUDACXX_HAS_NO_INT128
	static_assert(constexpr_test<__uint128_t>(),        "");
#endif


	runtime_test<unsigned char>();
	runtime_test<unsigned>();
	runtime_test<unsigned short>();
	runtime_test<unsigned long>();
	runtime_test<unsigned long long>();

	runtime_test<uint8_t>();
	runtime_test<uint16_t>();
	runtime_test<uint32_t>();
	runtime_test<uint64_t>();
	runtime_test<size_t>();
	runtime_test<uintmax_t>();
	runtime_test<uintptr_t>();

#ifndef _LIBCUDACXX_HAS_NO_INT128
	runtime_test<__uint128_t>();

	{
	__uint128_t val = 128;

	val <<= 32;
	assert( cuda::std::countr_one(val-1) == 39);
	assert( cuda::std::countr_one(val)   ==  0);
	assert( cuda::std::countr_one(val+1) ==  1);
	val <<= 2;
	assert( cuda::std::countr_one(val-1) == 41);
	assert( cuda::std::countr_one(val)   ==  0);
	assert( cuda::std::countr_one(val+1) ==  1);
	val <<= 3;
	assert( cuda::std::countr_one(val-1) == 44);
	assert( cuda::std::countr_one(val)   ==  0);
	assert( cuda::std::countr_one(val+1) ==  1);
	}
#endif

}
