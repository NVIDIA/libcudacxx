#include <stdio.h>
// We use <stdio.h> instead of <iostream> to avoid relying on the host system's
// C++ standard library.

__managed__ int ret;

__host__ __device__
int fake_main(int, char**);

__global__
void fake_main_kernel()
{
    ret = fake_main(0, NULL);
}

int main(int argc, char** argv)
{
    // Check if the CUDA driver/runtime are installed and working for sanity.
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA ERROR: %s: %s\n",
               cudaGetErrorName(err), cudaGetErrorString(err));
        return err;
    }

    ret = fake_main(argc, argv);
    if (ret != 0)
    {
        return ret;
    }

    fake_main_kernel<<<1, 1>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA ERROR: %s: %s\n",
               cudaGetErrorName(err), cudaGetErrorString(err));
        return err;
    }
    return ret;
}

#define main fake_main

