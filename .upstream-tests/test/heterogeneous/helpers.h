//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INTEROP_HELPERS_H
#define INTEROP_HELPERS_H

#include <new>
#include <stdlib.h>

#define INTEROP_SAFE_CALL(...) \
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

template<typename T>
__global__
void construct_kernel(void * address)
{
    new (address) T;
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

template<typename T>
T * device_default_initialize(void * address)
{
    construct_kernel<T><<<1, 1>>>(address);
    INTEROP_SAFE_CALL(cudaGetLastError());
    INTEROP_SAFE_CALL(cudaDeviceSynchronize());
    return reinterpret_cast<T *>(address);
}

template<typename T>
void device_destroy(T * object)
{
    destroy_kernel<<<1, 1>>>(object);
    INTEROP_SAFE_CALL(cudaGetLastError());
    INTEROP_SAFE_CALL(cudaDeviceSynchronize());
}

template<typename Tester, typename T>
void device_initialize(T & object)
{
    initialization_kernel<Tester><<<1, 1>>>(object);
    INTEROP_SAFE_CALL(cudaGetLastError());
    INTEROP_SAFE_CALL(cudaDeviceSynchronize());
}

template<typename Tester, typename T>
void device_validate(T & object)
{
    validation_kernel<Tester><<<1, 1>>>(object);
    INTEROP_SAFE_CALL(cudaGetLastError());
    INTEROP_SAFE_CALL(cudaDeviceSynchronize());
}

template<typename T>
using creator = T & (*)();

template<typename T>
using performer = void (*)(T &);

template<typename T>
struct initializer_validator
{
    performer<T> initializer;
    performer<T> validator;
};

template<typename T, typename ...Testers>
void validate_device_dynamic(tester_list<Testers...>)
{
    void * pointer;
    INTEROP_SAFE_CALL(cudaMalloc(&pointer, sizeof(T)));

    T & object = *device_default_initialize<T>(pointer);

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
    INTEROP_SAFE_CALL(cudaFree(pointer));
}

#if __cplusplus >= 201402L
template<typename T>
struct manual_object
{
    void construct() { new (static_cast<void *>(&data.object)) T; }
    void device_construct() { device_default_initialize<T>(&data.object); }

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

template<typename T>
__host__ __device__
manual_object<T> & get_managed()
{
    return managed_variable<T>;
}
#endif

template<typename T, std::size_t N>
void validate_in_managed_memory_helper(
    creator<T> creator_,
    performer<T> destroyer,
    initializer_validator<T> (&performers)[N]
)
{
    T & object = creator_();

    for (auto && performer : performers)
    {
        performer.initializer(object);
        performer.validator(object);
    }

    destroyer(object);
}

template<typename T, typename ...Testers>
void validate_managed(tester_list<Testers...>)
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

    auto host_constructor = []() -> T & {
        void * pointer;
        INTEROP_SAFE_CALL(cudaMallocManaged(&pointer, sizeof(T)));
        return *new (pointer) T;
    };

    auto device_constructor = []() -> T & {
        void * pointer;
        INTEROP_SAFE_CALL(cudaMallocManaged(&pointer, sizeof(T)));
        return *device_default_initialize<T>(pointer);
    };

    auto host_destructor = [](T & object){
        object.~T();
        INTEROP_SAFE_CALL(cudaFree(&object));
    };

    auto device_destructor = [](T & object){
        device_destroy(&object);
        INTEROP_SAFE_CALL(cudaFree(&object));
    };

    validate_in_managed_memory_helper<T>(host_constructor, host_destructor, host_init_device_check);
    validate_in_managed_memory_helper<T>(host_constructor, host_destructor, device_init_host_check);
    validate_in_managed_memory_helper<T>(host_constructor, device_destructor, host_init_device_check);
    validate_in_managed_memory_helper<T>(host_constructor, device_destructor, device_init_host_check);
    validate_in_managed_memory_helper<T>(device_constructor, host_destructor, host_init_device_check);
    validate_in_managed_memory_helper<T>(device_constructor, host_destructor, device_init_host_check);
    validate_in_managed_memory_helper<T>(device_constructor, device_destructor, host_init_device_check);
    validate_in_managed_memory_helper<T>(device_constructor, device_destructor, device_init_host_check);

#if __cplusplus >= 201402L
    auto host_variable_constructor = []() -> T & {
        get_managed<T>().construct();
        return get_managed<T>().get();
    };

    auto device_variable_constructor = []() -> T & {
        get_managed<T>().device_construct();
        return get_managed<T>().get();
    };

    auto host_variable_destructor = [](T &){
        get_managed<T>().destroy();
    };

    auto device_variable_destructor = [](T &){
        get_managed<T>().device_destroy();
    };

    validate_in_managed_memory_helper<T>(host_variable_constructor, host_variable_destructor, host_init_device_check);
    validate_in_managed_memory_helper<T>(host_variable_constructor, host_variable_destructor, device_init_host_check);
    validate_in_managed_memory_helper<T>(host_variable_constructor, device_variable_destructor, host_init_device_check);
    validate_in_managed_memory_helper<T>(host_variable_constructor, device_variable_destructor, device_init_host_check);
    validate_in_managed_memory_helper<T>(device_variable_constructor, host_variable_destructor, host_init_device_check);
    validate_in_managed_memory_helper<T>(device_variable_constructor, host_variable_destructor, device_init_host_check);
    validate_in_managed_memory_helper<T>(device_variable_constructor, device_variable_destructor, host_init_device_check);
    validate_in_managed_memory_helper<T>(device_variable_constructor, device_variable_destructor, device_init_host_check);
#endif
}

template<typename T, typename TesterList>
void validate_not_movable()
{
    TesterList list;

    validate_device_dynamic<T>(list);
    validate_managed<T>(list);
}

#endif
