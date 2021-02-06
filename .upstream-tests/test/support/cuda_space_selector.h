//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstddef>

#if defined(__CUDACC__) && !defined(__NVCOMPILER)
#define __exec_check_disable__ #pragma nv_exec_check_disable
#else
#define __exec_check_disable__
#endif

#ifdef __CUDACC_RTC__
#define LAMBDA [=]
#else
#define LAMBDA [=] __host__ __device__
#endif

#ifdef _LIBCUDACXX_CUDA_ARCH_DEF
#define SHARED __shared__
#else
#define SHARED
#endif

template<typename T, cuda::std::size_t SharedOffset>
struct malloc_memory_provider {
    static const constexpr cuda::std::size_t prefix_size
        = SharedOffset % alignof(T *) == 0
            ? SharedOffset
            : SharedOffset + (alignof(T *) - SharedOffset % alignof(T *));
    static const constexpr cuda::std::size_t shared_offset = prefix_size + sizeof(T *);

private:
    __host__ __device__
    T *& get_pointer() {
        alignas(T*)
        _LIBCUDACXX_CUDA_DISPATCH(
            DEVICE, _LIBCUDACXX_ARCH_BLOCK(
                __shared__ char storage[shared_offset];
                T *& allocated = *reinterpret_cast<T **>(storage + prefix_size);
                return allocated;
            ),
            HOST, _LIBCUDACXX_ARCH_BLOCK(
                static char storage[shared_offset];
                T *& allocated = *reinterpret_cast<T **>(storage + prefix_size);
                return allocated;
            )
        )
    }

public:
    __host__ __device__
    T * get() {
        _LIBCUDACXX_CUDA_DISPATCH(
            DEVICE, _LIBCUDACXX_ARCH_BLOCK(
                if (threadIdx.x == 0) {
                    get_pointer() = reinterpret_cast<T *>(malloc(sizeof(T) + alignof(T)));
                }
                __syncthreads();
            )
            HOST, _LIBCUDACXX_ARCH_BLOCK(
                get_pointer() = reinterpret_cast<T *>(malloc(sizeof(T) + alignof(T)));
            )
        )

        auto ptr = reinterpret_cast<cuda::std::uintptr_t>(get_pointer());
        ptr += alignof(T) - ptr % alignof(T);
        return reinterpret_cast<T *>(ptr);
    }

    __host__ __device__
    ~malloc_memory_provider() {
        _LIBCUDACXX_CUDA_DISPATCH(
            DEVICE, _LIBCUDACXX_ARCH_BLOCK(
                if (threadIdx.x == 0) {
                    free((void*)get_pointer());
                }
                __syncthreads();
            ),
            HOST, _LIBCUDACXX_ARCH_BLOCK(
                free((void*)get_pointer());
            )
        )
    }
};

template<typename T, cuda::std::size_t SharedOffset>
struct local_memory_provider {
    static const constexpr cuda::std::size_t prefix_size = 0;
    static const constexpr cuda::std::size_t shared_offset = SharedOffset;

    alignas(T) char buffer[sizeof(T)];

    __host__ __device__
    T * get() {
        return reinterpret_cast<T *>(&buffer);
    }
};

template<typename T, cuda::std::size_t SharedOffset>
struct device_shared_memory_provider {
    static const constexpr cuda::std::size_t prefix_size
        = SharedOffset % alignof(T) == 0
            ? SharedOffset
            : SharedOffset + (alignof(T) - SharedOffset % alignof(T));
    static const constexpr cuda::std::size_t shared_offset = prefix_size + sizeof(T);

    __device__
    T * get() {
        __shared__ alignas(T) char buffer[shared_offset];
        return reinterpret_cast<T *>(buffer + prefix_size);
    }
};

struct init_initializer {
    template<typename T, typename ...Ts>
    __host__ __device__
    static void construct(T & t, Ts && ...ts) {
        t.init(std::forward<Ts>(ts)...);
    }
};

struct constructor_initializer {
    template<typename T, typename ...Ts>
    __host__ __device__
    static void construct(T & t, Ts && ...ts) {
        new ((void*)&t) T(std::forward<Ts>(ts)...);
    }
};

struct default_initializer {
    template<typename T>
    __host__ __device__
    static void construct(T & t) {
        new ((void*)&t) T;
    }
};

template<typename T,
    template<typename, cuda::std::size_t> typename Provider,
    typename Initializer = constructor_initializer,
    cuda::std::size_t SharedOffset = 0>
class memory_selector
{
    Provider<T, SharedOffset> provider;
    T * ptr;

public:
    template<cuda::std::size_t SharedOffset_>
    using offsetted = memory_selector<T, Provider, Initializer, SharedOffset + SharedOffset_>;

    static const constexpr cuda::std::size_t prefix_size = Provider<T, SharedOffset>::prefix_size;
    static const constexpr cuda::std::size_t shared_offset = Provider<T, SharedOffset>::shared_offset;

#ifndef __CUDACC_RTC__
    __exec_check_disable__
#endif
    template<typename ...Ts>
    __host__ __device__
    T * construct(Ts && ...ts) {
        ptr = provider.get();
        assert(ptr);

        _LIBCUDACXX_CUDA_DISPATCH(
            DEVICE, _LIBCUDACXX_ARCH_BLOCK(
                if (threadIdx.x == 0) {
                    Initializer::construct(*ptr, std::forward<Ts>(ts)...);
                }
                __syncthreads();
            ),
            HOST, _LIBCUDACXX_ARCH_BLOCK(
                Initializer::construct(*ptr, std::forward<Ts>(ts)...);
            )
        )

        return ptr;
    }

#ifndef __CUDACC_RTC__
    __exec_check_disable__
#endif
    __host__ __device__
    ~memory_selector() {
        _LIBCUDACXX_CUDA_DISPATCH(
            DEVICE, _LIBCUDACXX_ARCH_BLOCK(
                if (threadIdx.x == 0) {
                    ptr->~T();
                }
                __syncthreads();
            ),
            HOST, _LIBCUDACXX_ARCH_BLOCK(
                ptr->~T();
            )
        )
    }
};

template<typename T, typename Initializer = constructor_initializer>
using local_memory_selector = memory_selector<T, local_memory_provider, Initializer>;

template<typename T, typename Initializer = constructor_initializer>
using shared_memory_selector = memory_selector<T, device_shared_memory_provider, Initializer>;

template<typename T, typename Initializer = constructor_initializer>
using global_memory_selector = memory_selector<T, malloc_memory_provider, Initializer>;
