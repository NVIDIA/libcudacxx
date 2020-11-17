//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
// We use <stdio.h> instead of <iostream> to avoid relying on the host system's
// C++ standard library.

void list_devices()
{
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("CUDA devices found: %d\n", device_count);

    int selected_device;
    cudaGetDevice(&selected_device);

    for (int dev = 0; dev < device_count; ++dev)
    {
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, dev);

        printf("Device %d: \"%s\", ", dev, device_prop.name);
        if(dev == selected_device)
            printf("Selected, ");
        else
            printf("Unused, ");

        printf("SM%d%d, %zu [bytes]\n",
            device_prop.major, device_prop.minor,
            device_prop.totalGlobalMem);
    }
}


__host__ __device__
int fake_main(int, char**);

int cuda_thread_count = 1;

__global__
void fake_main_kernel(int * ret)
{
    *ret = fake_main(0, NULL);
}

#define CUDA_CALL(err, ...) \
    do { \
        err = __VA_ARGS__; \
        if (err != cudaSuccess) \
        { \
            printf("CUDA ERROR, line %d: %s: %s\n", __LINE__,\
                   cudaGetErrorName(err), cudaGetErrorString(err)); \
            std::exit(1); \
        } \
    } while (false)

int main(int argc, char** argv)
{
    // Check if the CUDA driver/runtime are installed and working for sanity.
    cudaError_t err;
    CUDA_CALL(err, cudaDeviceSynchronize());

    list_devices();

    int ret = fake_main(argc, argv);
    if (ret != 0)
    {
        return ret;
    }

    int * cuda_ret = nullptr;
    CUDA_CALL(err, cudaMalloc(&cuda_ret, sizeof(int)));

    fake_main_kernel<<<1, cuda_thread_count>>>(cuda_ret);
    CUDA_CALL(err, cudaGetLastError());
    CUDA_CALL(err, cudaDeviceSynchronize());
    CUDA_CALL(err, cudaMemcpy(&ret, cuda_ret, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CALL(err, cudaFree(cuda_ret));

    return ret;
}

#define main fake_main

