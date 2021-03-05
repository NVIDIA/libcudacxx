#include <nv/target>
#include <stdio.h>

template <typename ...>
__host__ __device__ void test_comma() {}


#define OBSCURE_HOST_FEATURE 1
#if defined(OBSCURE_HOST_FEATURE)
#  define MY_QUERY NV_ANY_TARGET
#else
#  define MY_QUERY NV_IS_DEVICE
#endif

__host__ __device__ void test() {
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE,       (printf("device "); printf("invoked"); printf("\n"); test_comma<short, int, float>();),
    NV_IS_EXACTLY_SM80, (printf("==sm80 "); printf("invoked"); printf("\n"); test_comma<short, int, float>();),
    NV_PROVIDES_SM60,   (printf(">=sm60 "); printf("invoked"); printf("\n"); test_comma<short, int, float>();),
    NV_IS_HOST,         printf("host ");   printf("invoked"); printf("\n");
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
    (printf("!=sm80 "); printf("invoked"); printf("\n");)
  )

  NV_IF_TARGET(
    NV_ANY_TARGET,
    (printf("Any invoked\n"); test_comma<short, int, float>();),
    (printf("Should never be invoked\n");)
  )


  NV_IF_TARGET(
    NV_NO_TARGET,
    (printf("Should never be invoked\n");),
    (printf("No target else invoked\n"); test_comma<short, int, float>();)
  )


  NV_IF_TARGET(
    MY_QUERY,
    (
      NV_IF_TARGET(
        NV_IS_HOST, printf("Host bonus feature thing\n");
      )
      printf("Should be invoked\n");
    ),
    (printf("Query no host support\n"); test_comma<short, int, float>();)
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
