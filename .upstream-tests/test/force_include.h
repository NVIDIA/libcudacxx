#include <stdio.h>
// We use <stdio.h> instead of <iostream> to avoid relying on the host system's
// C++ standard library.

void list_devices()
{
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("CUDA devices found: %d.\n", device_count);

    int selected_device;
    cudaGetDevice(&selected_device);

    for (int dev = 0; dev < device_count; ++dev)
    {
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, dev);

        printf("Device %d: \"%s\"", dev, device_prop.name);
        if(dev == selected_device)
            printf(" [SELECTED]");

        printf(" (version: %d.%d, DRAM: %ld bytes)\n",
            device_prop.major, device_prop.minor,
            device_prop.totalGlobalMem);
    }
}


__host__ __device__
int fake_main(int, char**);

__global__
void fake_main_kernel(int * ret)
{
    *ret = fake_main(0, NULL);
}

#define CUDA_CALL(...) \
    do { \
        err = __VA_ARGS__; \
        if (err != cudaSuccess) \
        { \
            printf("CUDA ERROR: %s: %s\n", \
                   cudaGetErrorName(err), cudaGetErrorString(err)); \
            return err; \
        } \
    } while (false)

int main(int argc, char** argv)
{
    // Check if the CUDA driver/runtime are installed and working for sanity.
    cudaError_t err;
    CUDA_CALL(cudaDeviceSynchronize());

    list_devices();

    int ret = fake_main(argc, argv);
    if (ret != 0)
    {
        return ret;
    }

    int * cuda_ret = nullptr;
    CUDA_CALL(cudaMalloc(&cuda_ret, sizeof(int)));

    fake_main_kernel<<<1, 1>>>(cuda_ret);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaMemcpy(&ret, cuda_ret, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(cuda_ret));

    return ret;
}

#define main fake_main

