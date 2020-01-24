//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HETEROGENEOUS_HELPERS_H
#define HETEROGENEOUS_HELPERS_H

#include <new>
#include <stdlib.h>

#define HETEROGENEOUS_SAFE_CALL(...) \
    do { \
        cudaError_t err = __VA_ARGS__; \
        if (err != cudaSuccess) \
        { \
            printf("CUDA ERROR: %s: %s\n", \
                cudaGetErrorName(err), cudaGetErrorString(err)); \
            abort(); \
        } \
    } while (false)

template<typename ...Testers>
struct tester_list
{
};

template<typename TesterList, typename ...Testers>
struct extend_tester_list_t;

template<typename ...Original, typename ...Additional>
struct extend_tester_list_t<tester_list<Original...>, Additional...>
{
    using type = tester_list<Original..., Additional...>;
};

template<typename TesterList, typename ...Testers>
using extend_tester_list = typename extend_tester_list_t<TesterList, Testers...>::type;

template<typename Tester, typename T>
__host__ __device__
void initialize(T & object)
{
    Tester::initialize(object);
}

template<typename Tester, typename T>
__host__ __device__
auto validate_impl(T & object)
    -> decltype(Tester::validate(object), void())
{
    Tester::validate(object);
}

template<typename, typename ...Ts>
__host__ __device__
void validate_impl(Ts && ...)
{
}

template<typename Tester, typename T>
__host__ __device__
void validate(T & object)
{
    validate_impl<Tester>(object);
}

template<typename T, typename ...Args>
__global__
void construct_kernel(void * address, Args ...args)
{
    new (address) T(args...);
}

template<typename T>
__global__
void destroy_kernel(T * object)
{
    object->~T();
}

template<typename Tester, typename T>
__global__
void initialization_kernel(T & object)
{
    initialize<Tester>(object);
}

template<typename Tester, typename T>
__global__
void validation_kernel(T & object)
{
    validate<Tester>(object);
}

template<typename T, typename ...Args>
T * device_construct(void * address, Args... args)
{
    construct_kernel<T><<<1, 1>>>(address, args...);
    HETEROGENEOUS_SAFE_CALL(cudaGetLastError());
    HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());
    return reinterpret_cast<T *>(address);
}

template<typename T>
void device_destroy(T * object)
{
    destroy_kernel<<<1, 1>>>(object);
    HETEROGENEOUS_SAFE_CALL(cudaGetLastError());
    HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());
}

template<typename Tester, typename T>
void device_initialize(T & object)
{
    initialization_kernel<Tester><<<1, 1>>>(object);
    HETEROGENEOUS_SAFE_CALL(cudaGetLastError());
    HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());
}

template<typename Tester, typename T>
void device_validate(T & object)
{
    validation_kernel<Tester><<<1, 1>>>(object);
    HETEROGENEOUS_SAFE_CALL(cudaGetLastError());
    HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());
}

template<typename T, typename ...Args>
using creator = T & (*)(Args...);

template<typename T>
using performer = void (*)(T &);

template<typename T>
struct initializer_validator
{
    performer<T> initializer;
    performer<T> validator;
};

template<typename T, typename ...Testers, typename ...Args>
void validate_device_dynamic(tester_list<Testers...>, Args ...args)
{
    void * pointer;
    HETEROGENEOUS_SAFE_CALL(cudaMalloc(&pointer, sizeof(T)));

    T & object = *device_construct<T>(pointer, args...);

    initializer_validator<T> performers[] = {
        {
            device_initialize<Testers>,
            device_validate<Testers>
        }...
    };

    for (auto && performer : performers)
    {
        performer.initializer(object);
        performer.validator(object);
    }

    device_destroy(&object);
    HETEROGENEOUS_SAFE_CALL(cudaFree(pointer));
}

#if __cplusplus >= 201402L
template<typename T>
struct manual_object
{
    template<typename ...Args>
    void construct(Args ...args) { new (static_cast<void *>(&data.object)) T(args...); }
    template<typename ...Args>
    void device_construct(Args ...args) { ::device_construct<T>(&data.object, args...); }

    void destroy() { data.object.~T(); }
    void device_destroy() { ::device_destroy(&data.object); }

    T & get() { return data.object; }

    union data
    {
        __host__ __device__
        data() {}

        __host__ __device__
        ~data() {}

        T object;
    } data;
};

template<typename T>
__managed__ manual_object<T> managed_variable;
#endif

template<typename T, std::size_t N, typename ...Args>
void validate_in_managed_memory_helper(
    creator<T, Args...> creator_,
    performer<T> destroyer,
    initializer_validator<T> (&performers)[N],
    Args ...args
)
{
    T & object = creator_(args...);

    for (auto && performer : performers)
    {
        performer.initializer(object);
        performer.validator(object);
    }

    destroyer(object);
}

template<typename T, typename ...Testers, typename ...Args>
void validate_managed(tester_list<Testers...>, Args ...args)
{
    initializer_validator<T> host_init_device_check[] = {
        {
            initialize<Testers>,
            device_validate<Testers>
        }...
    };

    initializer_validator<T> device_init_host_check[] = {
        {
            device_initialize<Testers>,
            validate<Testers>
        }...
    };

    creator<T, Args...> host_constructor = [](Args ...args) -> T & {
        void * pointer;
        HETEROGENEOUS_SAFE_CALL(cudaMallocManaged(&pointer, sizeof(T)));
        return *new (pointer) T(args...);
    };

    creator<T, Args...> device_constructor = [](Args ...args) -> T & {
        void * pointer;
        HETEROGENEOUS_SAFE_CALL(cudaMallocManaged(&pointer, sizeof(T)));
        return *device_construct<T>(pointer, args...);
    };

    performer<T> host_destructor = [](T & object){
        object.~T();
        HETEROGENEOUS_SAFE_CALL(cudaFree(&object));
    };

    performer<T> device_destructor = [](T & object){
        device_destroy(&object);
        HETEROGENEOUS_SAFE_CALL(cudaFree(&object));
    };

    validate_in_managed_memory_helper<T>(host_constructor, host_destructor, host_init_device_check, args...);
    validate_in_managed_memory_helper<T>(host_constructor, host_destructor, device_init_host_check, args...);
    validate_in_managed_memory_helper<T>(host_constructor, device_destructor, host_init_device_check, args...);
    validate_in_managed_memory_helper<T>(host_constructor, device_destructor, device_init_host_check, args...);
    validate_in_managed_memory_helper<T>(device_constructor, host_destructor, host_init_device_check, args...);
    validate_in_managed_memory_helper<T>(device_constructor, host_destructor, device_init_host_check, args...);
    validate_in_managed_memory_helper<T>(device_constructor, device_destructor, host_init_device_check, args...);
    validate_in_managed_memory_helper<T>(device_constructor, device_destructor, device_init_host_check, args...);

#if __cplusplus >= 201402L && !defined(__clang__)
    // The managed variable template part of this test is disabled under clang, pending nvbug 2790305 being fixed.

    creator<T, Args...> host_variable_constructor = [](Args ...args) -> T & {
        managed_variable<T>.construct(args...);
        return managed_variable<T>.get();
    };

    creator<T, Args...> device_variable_constructor = [](Args ...args) -> T & {
        managed_variable<T>.device_construct(args...);
        return managed_variable<T>.get();
    };

    performer<T> host_variable_destructor = [](T &){
        managed_variable<T>.destroy();
    };

    performer<T> device_variable_destructor = [](T &){
        managed_variable<T>.device_destroy();
    };

    validate_in_managed_memory_helper<T>(host_variable_constructor, host_variable_destructor, host_init_device_check, args...);
    validate_in_managed_memory_helper<T>(host_variable_constructor, host_variable_destructor, device_init_host_check, args...);
    validate_in_managed_memory_helper<T>(host_variable_constructor, device_variable_destructor, host_init_device_check, args...);
    validate_in_managed_memory_helper<T>(host_variable_constructor, device_variable_destructor, device_init_host_check, args...);
    validate_in_managed_memory_helper<T>(device_variable_constructor, host_variable_destructor, host_init_device_check, args...);
    validate_in_managed_memory_helper<T>(device_variable_constructor, host_variable_destructor, device_init_host_check, args...);
    validate_in_managed_memory_helper<T>(device_variable_constructor, device_variable_destructor, host_init_device_check, args...);
    validate_in_managed_memory_helper<T>(device_variable_constructor, device_variable_destructor, device_init_host_check, args...);
#endif
}

#define HELPERS_CUDA_CALL(err, ...) \
    do { \
        err = __VA_ARGS__; \
        if (err != cudaSuccess) \
        { \
            printf("CUDA ERROR, line %d: %s: %s\n", __LINE__,\
                   cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while (false)

bool check_managed_memory_support()
{
    int current_device, property_value;
    cudaError_t err;
    HELPERS_CUDA_CALL(err, cudaGetDevice(&current_device));
    HELPERS_CUDA_CALL(err, cudaDeviceGetAttribute(&property_value, cudaDevAttrManagedMemory, current_device));
    return property_value == 1;
}

struct dummy_tester
{
    template<typename ...Ts>
    __host__ __device__
    static void initialize(Ts &&...) {}

    template<typename ...Ts>
    __host__ __device__
    static void validate(Ts &&...) {}
};

template<typename List>
struct validate_list
{
    using type = List;
};

template<>
struct validate_list<tester_list<>>
{
    using type = tester_list<dummy_tester>;
};

template<typename T, typename TesterList, typename ...Args>
void validate_not_movable(Args ...args)
{
    typename validate_list<TesterList>::type list;

    validate_device_dynamic<T>(list, args...);
    if (check_managed_memory_support())
    {
        validate_managed<T>(list, args...);
    }
}

#endif
