//===----------------------------------------------------------------------===//
//
// Part of the CUDA Toolkit, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CUDACC__
#define __exec_check_disable__ #pragma nv_exec_check_disable
#else
#define __exec_check_disable__
#endif

#ifdef __CUDACC_RTC__
#define LAMBDA [=]
#else
#define LAMBDA [=] __host__ __device__
#endif

#ifdef __CUDA_ARCH__
#define SHARED __shared__
#else
#define SHARED
#endif

template<typename T>
struct malloc_memory_provider {
private:
    __host__ __device__
    T *& get_pointer() {
#ifdef __CUDA_ARCH__
        __shared__
#else
        static
#endif
        T * allocated;
        return allocated;
    }

public:
    __host__ __device__
    T * get() {
#ifdef __CUDA_ARCH__
        if (threadIdx.x == 0) {
#endif
        get_pointer() = reinterpret_cast<T *>(malloc(sizeof(T)));
#ifdef __CUDA_ARCH__
        }
        __syncthreads();
#endif
        return get_pointer();
    }

    __host__ __device__
    ~malloc_memory_provider() {
#ifdef __CUDA_ARCH__
        if (threadIdx.x == 0) {
#endif
        free((void*)get_pointer());
#ifdef __CUDA_ARCH__
        }
        __syncthreads();
#endif
    }
};

template<typename T>
struct local_memory_provider {
    alignas(T) char buffer[sizeof(T)];

    __host__ __device__
    T * get() {
        return reinterpret_cast<T *>(&buffer);
    }
};

template<typename T>
struct device_shared_memory_provider {
    __device__
    T * get() {
        __shared__ char buffer[sizeof(T)];
        return reinterpret_cast<T *>(&buffer);
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
    template<typename> typename Provider,
    typename Initializer = constructor_initializer>
class memory_selector
{
    Provider<T> provider;
    T * ptr;

public:
#ifndef __CUDACC_RTC__
    __exec_check_disable__
#endif
    template<typename ...Ts>
    __host__ __device__
    T * construct(Ts && ...ts) {
        ptr = provider.get();
        assert(ptr);
#ifdef __CUDA_ARCH__
        if (threadIdx.x == 0) {
#endif
        Initializer::construct(*ptr, std::forward<Ts>(ts)...);
#ifdef __CUDA_ARCH__
        }
        __syncthreads();
#endif
        return ptr;
    }

#ifndef __CUDACC_RTC__
    __exec_check_disable__
#endif
    __host__ __device__
    ~memory_selector() {
#ifdef __CUDA_ARCH__
        if (threadIdx.x == 0) {
#endif
        ptr->~T();
#ifdef __CUDA_ARCH__
        }
        __syncthreads();
#endif
    }
};

template<typename T, typename Initializer = constructor_initializer>
using local_memory_selector = memory_selector<T, local_memory_provider, Initializer>;

template<typename T, typename Initializer = constructor_initializer>
using shared_memory_selector = memory_selector<T, device_shared_memory_provider, Initializer>;

template<typename T, typename Initializer = constructor_initializer>
using global_memory_selector = memory_selector<T, malloc_memory_provider, Initializer>;
