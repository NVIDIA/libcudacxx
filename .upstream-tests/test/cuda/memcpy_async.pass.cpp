//===----------------------------------------------------------------------===//
//
// Part of the CUDA Toolkit, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70

#include <cuda/barrier>

#include "cuda_space_selector.h"

struct large_type
{
    constexpr static int size = 32;

    __host__ __device__
    large_type(int val = 0) {
        for (cuda::std::size_t i = 0; i < size; ++i) {
            storage[i] = val;
        }
    }

    large_type(const large_type &) = default;
    large_type & operator=(const large_type &) = default;

    __host__ __device__
    friend bool operator==(const large_type & lhs, const large_type & rhs) {
        for (cuda::std::size_t i = 0; i < size; ++i) {
            if (lhs.storage[i] != rhs.storage[i]) {
                return false;
            }
        }

        return true;
    }

    int storage[size];
};

template <class T,
    template<typename, typename> class SourceSelector,
    template<typename, typename> class DestSelector,
    template<typename, typename> class BarrierSelector
>
__host__ __device__ __noinline__
void test_fully_specialized()
{
    SourceSelector<T, constructor_initializer> source_sel;
    typename DestSelector<T, constructor_initializer>
        ::template offsetted<decltype(source_sel)::shared_offset> dest_sel;
    BarrierSelector<cuda::barrier<cuda::thread_scope_block>, constructor_initializer> bar_sel;

    T * source = source_sel.construct(static_cast<T>(12));
    T * dest = dest_sel.construct(static_cast<T>(0));
    cuda::barrier<cuda::thread_scope_block> * bar = bar_sel.construct(1);

    assert(*source == 12);
    assert(*dest == 0);

    cuda::memcpy_async(dest, source, sizeof(T), *bar);

    bar->arrive_and_wait();

    assert(*source == 12);
    assert(*dest == 12);

    *source = 24;

    cuda::memcpy_async(static_cast<void *>(dest), static_cast<void *>(source), sizeof(T), *bar);

    bar->arrive_and_wait();

    assert(*source == 24);
    assert(*dest == 24);
}

template <class T,
    template<typename, typename> class SourceSelector,
    template<typename, typename> class DestSelector
>
__host__ __device__ __noinline__
void test_select_barrier()
{
    test_fully_specialized<T, SourceSelector, DestSelector, local_memory_selector>();
#ifdef __CUDA_ARCH__
    test_fully_specialized<T, SourceSelector, DestSelector, shared_memory_selector>();
    test_fully_specialized<T, SourceSelector, DestSelector, global_memory_selector>();
#endif
}

template <class T, template<typename, typename> class SourceSelector>
struct TestFn {
    __host__ __device__ __noinline__
    void operator()() const
    {
        test_select_barrier<T, SourceSelector, local_memory_selector>();
#ifdef __CUDA_ARCH__
        test_select_barrier<T, SourceSelector, shared_memory_selector>();
        test_select_barrier<T, SourceSelector, global_memory_selector>();
#endif
    }
};

template <template <class, template<typename, typename> class> class TestFunctor,
    template<typename, typename> class Selector
>
struct TestEachType {
    __host__ __device__ __noinline__
    void operator()() const {
        TestFunctor<  int8_t, Selector>()();
        TestFunctor<uint16_t, Selector>()();
        TestFunctor< int32_t, Selector>()();
        TestFunctor<uint64_t, Selector>()();
        TestFunctor<large_type, Selector>()();
    }
};

int main(int argc, char ** argv)
{
    TestEachType<TestFn, local_memory_selector>()();
#ifdef __CUDA_ARCH__
    TestEachType<TestFn, shared_memory_selector>()();
    TestEachType<TestFn, global_memory_selector>()();
#endif

    return 0;
}
