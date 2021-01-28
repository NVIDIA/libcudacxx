#include <cuda/std/version>
#include <stdio.h>

__host__ __device__ void test() {
  _LIBCUDACXX_CUDA_DISPATCH(
      DEVICE,                  _LIBCUDACXX_ARCH_INVOKE([](){printf("device invoked\n");}),
      SM80,                    _LIBCUDACXX_ARCH_INVOKE([](){printf("sm80 invoked\n");}),
      LESS_THAN_SM80,          _LIBCUDACXX_ARCH_INVOKE([](){printf("=<sm70 invoked\n");}),
      GREATER_THAN_SM70,       _LIBCUDACXX_ARCH_INVOKE([](){printf(">sm70 invoked\n");}),
      HOST,                    _LIBCUDACXX_ARCH_INVOKE([](){printf("host invoked\n");})
  )

  _LIBCUDACXX_CUDA_DISPATCH(
      DEVICE,                  _LIBCUDACXX_ARCH_BLOCK(printf("device "); printf("invoked"); printf("\n");),
      SM80,                    _LIBCUDACXX_ARCH_BLOCK(printf("sm80 "); printf("invoked"); printf("\n");),
      LESS_THAN_SM80,          _LIBCUDACXX_ARCH_BLOCK(printf("=<sm70 "); printf("invoked"); printf("\n");),
      GREATER_THAN_SM70,       _LIBCUDACXX_ARCH_BLOCK(printf(">sm70 "); printf("invoked"); printf("\n");),
      HOST,                    _LIBCUDACXX_ARCH_BLOCK(printf("host "); printf("invoked"); printf("\n");)
  )
}

__global__ void launch() {
  test();
}

int main() {
    test();
    launch<<<1,1>>>();
    cudaDeviceSynchronize();
}
