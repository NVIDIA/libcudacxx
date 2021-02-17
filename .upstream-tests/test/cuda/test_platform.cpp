#include <nv/target>
#include <stdio.h>

template <typename ...>
__host__ __device__ void test_comma() {}

__host__ __device__ void test() {
  NV_DISPATCH_TARGET(
      NV_IS_DEVICE,       (printf("device "); printf("invoked"); printf("\n"); test_comma<short, int, float>();),
      NV_IS_EXACTLY_SM80, (printf("==sm80 "); printf("invoked"); printf("\n"); test_comma<short, int, float>();),
      NV_PROVIDES_SM60,   (printf(">=sm60 "); printf("invoked"); printf("\n"); test_comma<short, int, float>();),
      NV_IS_HOST,         (printf("host ");   printf("invoked"); printf("\n"); test_comma<short, int, float>();)
  )

  NV_DISPATCH_TARGET(
      NV_IS_EXACTLY_SM80, (printf("==sm80 "); printf("invoked"); printf("\n"); test_comma<short, int, float>();),
      NV_IS_DEVICE,       (printf("device "); printf("invoked"); printf("\n"); test_comma<short, int, float>();),
      NV_PROVIDES_SM60,   (printf(">=sm60 "); printf("invoked"); printf("\n"); test_comma<short, int, float>();)
  )

  NV_DISPATCH_TARGET(
      NV_PROVIDES_SM60,   (printf(">=sm60 "); printf("invoked"); printf("\n"); test_comma<short, int, float>();),
      NV_IS_EXACTLY_SM80, (printf("==sm80 "); printf("invoked"); printf("\n"); test_comma<short, int, float>();),
      NV_IS_DEVICE,       (printf("device "); printf("invoked"); printf("\n"); test_comma<short, int, float>();)
  )

  NV_IF_TARGET(
      NV_PROVIDES_SM60,
      (printf(">=sm60 "); printf("invoked"); printf("\n"); test_comma<short, int, float>();)
  )

  NV_IF_TARGET(
      NV_IS_EXACTLY_SM80,
      (printf("==sm80 "); printf("invoked"); printf("\n"); test_comma<short, int, float>();),
      (printf("!=sm80 "); printf("invoked"); printf("\n"); test_comma<short, int, float>();)
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
