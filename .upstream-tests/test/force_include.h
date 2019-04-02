__managed__ int ret;

__host__ __device__
int fake_main(int, char**);

__global__
void fake_main_kernel()
{
    ret = fake_main(0, NULL);
}

int main()
{
    ret = fake_main(0, NULL);
    if (ret != 0)
    {
        return ret;
    }

    fake_main_kernel<<<1, 1>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        return err;
    }
    return ret;
}

#define main fake_main

