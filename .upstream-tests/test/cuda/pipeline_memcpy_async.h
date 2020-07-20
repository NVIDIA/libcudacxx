//===----------------------------------------------------------------------===//
//
// Part of the CUDA Toolkit, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70

#include <cooperative_groups.h>
#include <cuda/pipeline>

#include "cuda_space_selector.h"
#include "large_type.h"

template <class T,
    template<typename, typename> class SourceSelector,
    template<typename, typename> class DestSelector,
    template<typename, typename> class PipelineSelector,
    cuda::thread_scope PipelineScope,
    uint8_t PipelineStages
>
__device__ __noinline__
void test_fully_specialized()
{
    SourceSelector<T, constructor_initializer> source_sel;
    typename DestSelector<T, constructor_initializer>
        ::template offsetted<decltype(source_sel)::shared_offset> dest_sel;
    PipelineSelector<cuda::pipeline_shared_state<PipelineScope, PipelineStages>, constructor_initializer> pipe_state_sel;

    T * source = source_sel.construct(static_cast<T>(12));
    T * dest = dest_sel.construct(static_cast<T>(0));
    cuda::pipeline_shared_state<PipelineScope, PipelineStages> * pipe_state = pipe_state_sel.construct();

    auto pipe = make_pipeline(cooperative_groups::this_thread_block(), pipe_state);

    assert(*source == 12);
    assert(*dest == 0);

    pipe.producer_acquire();
    cuda::memcpy_async(dest, source, sizeof(T), pipe);
    pipe.producer_commit();
    pipe.consumer_wait();

    assert(*source == 12);
    assert(*dest == 12);

    pipe.consumer_release();

    *source = 24;

    pipe.producer_acquire();
    cuda::memcpy_async(static_cast<void *>(dest), static_cast<void *>(source), sizeof(T), pipe);
    pipe.producer_commit();
    pipe.consumer_wait();

    assert(*source == 24);
    assert(*dest == 24);

    pipe.consumer_release();
}

template <class T,
    template<typename, typename> class SourceSelector,
    template<typename, typename> class DestSelector,
    template<typename, typename> class PipelineSelector,
    cuda::thread_scope PipelineScope
>
__host__ __device__ __noinline__
void test_select_stages()
{
    test_fully_specialized<T, SourceSelector, DestSelector, PipelineSelector, PipelineScope, 1>();
    test_fully_specialized<T, SourceSelector, DestSelector, PipelineSelector, PipelineScope, 8>();
}

template <class T,
    template<typename, typename> class SourceSelector,
    template<typename, typename> class DestSelector,
    template<typename, typename> class PipelineSelector
>
__host__ __device__ __noinline__
void test_select_scope()
{
#ifdef __CUDA_ARCH__
    test_select_stages<T, SourceSelector, DestSelector, PipelineSelector, cuda::thread_scope_block>();
    test_select_stages<T, SourceSelector, DestSelector, PipelineSelector, cuda::thread_scope_device>();
    test_select_stages<T, SourceSelector, DestSelector, PipelineSelector, cuda::thread_scope_system>();
#endif
}

template <class T,
    template<typename, typename> class SourceSelector,
    template<typename, typename> class DestSelector
>
__host__ __device__ __noinline__
void test_select_pipeline()
{
    test_select_scope<T, SourceSelector, DestSelector, local_memory_selector>();
#ifdef __CUDA_ARCH__
    test_select_scope<T, SourceSelector, DestSelector, shared_memory_selector>();
    test_select_scope<T, SourceSelector, DestSelector, global_memory_selector>();
#endif
}

template <class T,
    template<typename, typename> class SourceSelector
>
__host__ __device__ __noinline__
void test_select_destination()
{
    test_select_pipeline<T, SourceSelector, local_memory_selector>();
#ifdef __CUDA_ARCH__
    test_select_pipeline<T, SourceSelector, shared_memory_selector>();
    test_select_pipeline<T, SourceSelector, global_memory_selector>();
#endif
}

template <class T>
__host__ __device__ __noinline__
void test_select_source()
{
    test_select_destination<T, local_memory_selector>();
#ifdef __CUDA_ARCH__
    test_select_destination<T, shared_memory_selector>();
    test_select_destination<T, global_memory_selector>();
#endif
}
