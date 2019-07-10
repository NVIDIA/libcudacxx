//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include "helpers.h"

#include <cuda/std/atomic>

template<int Operand>
struct store_tester
{
    template<typename T>
    __host__ __device__
    static void initialize(cuda::std::atomic<T> & a)
    {
        a.store(Operand);
    }

    template<typename T>
    __host__ __device__
    static void validate(cuda::std::atomic<T> & a)
    {
        assert(a.load() == static_cast<T>(Operand));
    }
};

#define ATOMIC_TESTER(operation) \
    template<int PreviousValue, int Operand, int ExpectedValue> \
    struct operation ## _tester \
    { \
        template<typename T> \
        __host__ __device__ \
        static void initialize(cuda::std::atomic<T> & a) \
        { \
            assert(a.operation(Operand) == static_cast<T>(PreviousValue)); \
        } \
        \
        template<typename T> \
        __host__ __device__ \
        static void validate(cuda::std::atomic<T> & a) \
        { \
            assert(a.load() == static_cast<T>(ExpectedValue)); \
        } \
    };

ATOMIC_TESTER(fetch_add);
ATOMIC_TESTER(fetch_sub);

ATOMIC_TESTER(fetch_and);
ATOMIC_TESTER(fetch_or);
ATOMIC_TESTER(fetch_xor);

using store_atomic_testers = tester_list<
    store_tester<0>,
    store_tester<-1>,
    store_tester<17>
>;

using arithmetic_atomic_testers = extend_tester_list<
    store_atomic_testers,
    fetch_add_tester<17, 13, 30>,
    fetch_sub_tester<30, 21, 9>,
    fetch_sub_tester<9, 17, -8>
>;

using bitwise_atomic_testers = extend_tester_list<
    arithmetic_atomic_testers,
    fetch_add_tester<-8, 10, 2>,
    fetch_or_tester<2, 13, 15>,
    fetch_and_tester<15, 8, 8>,
    fetch_and_tester<8, 13, 8>,
    fetch_xor_tester<8, 12, 4>
>;

void kernel_invoker()
{
    validate_not_movable<cuda::std::atomic<signed char>, arithmetic_atomic_testers>();
    validate_not_movable<cuda::std::atomic<signed short>, arithmetic_atomic_testers>();
    validate_not_movable<cuda::std::atomic<signed int>, arithmetic_atomic_testers>();
    validate_not_movable<cuda::std::atomic<signed long>, arithmetic_atomic_testers>();
    validate_not_movable<cuda::std::atomic<signed long long>, arithmetic_atomic_testers>();

    validate_not_movable<cuda::std::atomic<unsigned char>, bitwise_atomic_testers>();
    validate_not_movable<cuda::std::atomic<unsigned short>, bitwise_atomic_testers>();
    validate_not_movable<cuda::std::atomic<unsigned int>, bitwise_atomic_testers>();
    validate_not_movable<cuda::std::atomic<unsigned long>, bitwise_atomic_testers>();
    validate_not_movable<cuda::std::atomic<unsigned long long>, bitwise_atomic_testers>();
}

int main(int arg, char ** argv)
{
#ifndef __CUDA_ARCH__
    kernel_invoker();
#endif

    return 0;
}
