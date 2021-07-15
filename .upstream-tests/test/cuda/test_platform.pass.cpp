//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <nv/target>

#if !defined(__CUDACC_RTC__)
#include <assert.h>
#include <stdio.h>
#endif

#if defined(__NVCC__) || defined(__CUDACC_RTC__)
#  define TEST_NVCC
#elif defined(__NVCOMPILER)
#  define TEST_NVCXX
#else
#  define TEST_HOST
#endif

#if defined(TEST_NVCC)

__host__ __device__ void test() {
#if defined(__CUDA_ARCH__)
  constexpr int arch_val = __CUDA_ARCH__;
#else
  constexpr int arch_val = 0;
#endif

  // This test ensures that the fallthrough cases are not invoked.
  // SM_80 would imply that SM_72 is available, yet it should not be expanded by the macro
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_86, (static_assert(arch_val >= 860, "cuda arch expected 860");),
    NV_PROVIDES_SM_80, (static_assert(arch_val >= 800, "cuda arch expected 800");),
    NV_PROVIDES_SM_75, (static_assert(arch_val >= 750, "cuda arch expected 750");),
    NV_PROVIDES_SM_72, (static_assert(arch_val >= 720, "cuda arch expected 720");),
    NV_PROVIDES_SM_70, (static_assert(arch_val >= 700, "cuda arch expected 700");),
    NV_PROVIDES_SM_62, (static_assert(arch_val >= 620, "cuda arch expected 620");),
    NV_PROVIDES_SM_61, (static_assert(arch_val >= 610, "cuda arch expected 610");),
    NV_PROVIDES_SM_60, (static_assert(arch_val >= 600, "cuda arch expected 600");),
    NV_PROVIDES_SM_53, (static_assert(arch_val >= 530, "cuda arch expected 530");),
    NV_PROVIDES_SM_52, (static_assert(arch_val >= 520, "cuda arch expected 520");),
    NV_PROVIDES_SM_50, (static_assert(arch_val >= 500, "cuda arch expected 500");),
    NV_PROVIDES_SM_37, (static_assert(arch_val >= 370, "cuda arch expected 370");),
    NV_PROVIDES_SM_35, (static_assert(arch_val >= 350, "cuda arch expected 350");),
    NV_IS_HOST,        (static_assert(arch_val == 0,   "cuda arch expected 0");)
  )

  // This test is simpler and ensures that only the value matched is invoked, but is roughly the same as the above
  NV_DISPATCH_TARGET(
    NV_IS_EXACTLY_SM_86, (static_assert(arch_val == 860, "cuda arch expected 800");),
    NV_IS_EXACTLY_SM_80, (static_assert(arch_val == 800, "cuda arch expected 800");),
    NV_IS_EXACTLY_SM_75, (static_assert(arch_val == 750, "cuda arch expected 750");),
    NV_IS_EXACTLY_SM_72, (static_assert(arch_val == 720, "cuda arch expected 720");),
    NV_IS_EXACTLY_SM_70, (static_assert(arch_val == 700, "cuda arch expected 700");),
    NV_IS_EXACTLY_SM_62, (static_assert(arch_val == 620, "cuda arch expected 620");),
    NV_IS_EXACTLY_SM_61, (static_assert(arch_val == 610, "cuda arch expected 610");),
    NV_IS_EXACTLY_SM_60, (static_assert(arch_val == 600, "cuda arch expected 600");),
    NV_IS_EXACTLY_SM_53, (static_assert(arch_val == 530, "cuda arch expected 530");),
    NV_IS_EXACTLY_SM_52, (static_assert(arch_val == 520, "cuda arch expected 520");),
    NV_IS_EXACTLY_SM_50, (static_assert(arch_val == 500, "cuda arch expected 500");),
    NV_IS_EXACTLY_SM_37, (static_assert(arch_val == 370, "cuda arch expected 370");),
    NV_IS_EXACTLY_SM_35, (static_assert(arch_val == 350, "cuda arch expected 350");),
    NV_IS_HOST,          (static_assert(arch_val == 0,   "cuda arch expected 0");)
  )

  NV_IF_TARGET(
    NV_IS_HOST,
      (static_assert(arch_val == 0, "cuda arch expected 0");),
      (static_assert(arch_val != 0, "cuda arch expected !0");)
  )

  // Some additional tests, but briefly exercise the parenthesis hacks on NVCC
  NV_IF_TARGET(
    NV_IS_DEVICE,
      static_assert(arch_val != 0, "cuda arch expected !0");,
      static_assert(arch_val == 0, "cuda arch expected 0");
  )

  NV_DISPATCH_TARGET(
    NV_IS_DEVICE, static_assert(arch_val != 0, "cuda arch expected !0");,
    NV_IS_HOST,   static_assert(arch_val == 0, "cuda arch expected 0");
  )

  NV_DISPATCH_TARGET(
    NV_NO_TARGET, assert("Should never be hit");,
    NV_ANY_TARGET, static_assert(arch_val == arch_val, "");
  )

  NV_IF_TARGET(
    NV_IS_HOST,
      printf("Host success\r\n");,
      printf("Device success\r\n");
  )

  NV_DISPATCH_TARGET(
    NV_IS_HOST, (),
    NV_IS_DEVICE, (
      static_assert(NV_TARGET_MINIMUM_SM_INTEGER == (__CUDA_ARCH__ / 10), "arch mismatch");
      static_assert(__CUDA_MINIMUM_ARCH__ == __CUDA_ARCH__, "arch mismatch");
    )
  )
}

#elif defined(TEST_NVCXX)

__host__ __device__ void test() {
  int invoke_count = 0;

  // This test ensures that the fallthrough cases are not invoked.
  // SM_80 would imply that SM_72 is available, yet it should not be expanded or invoked by the macro
  // Test accessing threadIdx.x to ensure that only device code is hitting those code paths
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_80,  invoke_count += 1; invoke_count += threadIdx.x;,
    NV_PROVIDES_SM_75,  invoke_count += 1; invoke_count += threadIdx.x;,
    NV_PROVIDES_SM_72,  invoke_count += 1; invoke_count += threadIdx.x;,
    NV_PROVIDES_SM_70,  invoke_count += 1; invoke_count += threadIdx.x;,
    NV_PROVIDES_SM_62,  invoke_count += 1; invoke_count += threadIdx.x;,
    NV_PROVIDES_SM_61,  invoke_count += 1; invoke_count += threadIdx.x;,
    NV_PROVIDES_SM_60, (invoke_count += 1; invoke_count += threadIdx.x;),
    NV_PROVIDES_SM_53, (invoke_count += 1; invoke_count += threadIdx.x;),
    NV_PROVIDES_SM_52, (invoke_count += 1; invoke_count += threadIdx.x;),
    NV_PROVIDES_SM_50, (invoke_count += 1; invoke_count += threadIdx.x;),
    NV_PROVIDES_SM_37, (invoke_count += 1; invoke_count += threadIdx.x;),
    NV_PROVIDES_SM_35, (invoke_count += 1; invoke_count += threadIdx.x;),
    NV_IS_HOST,        (invoke_count += 1;)
  )

  assert(invoke_count == 1);
  invoke_count = 0;

  NV_DISPATCH_TARGET(
    NV_IS_EXACTLY_SM_80,  invoke_count += 1; invoke_count += threadIdx.x;,
    NV_IS_EXACTLY_SM_75,  invoke_count += 1; invoke_count += threadIdx.x;,
    NV_IS_EXACTLY_SM_72,  invoke_count += 1; invoke_count += threadIdx.x;,
    NV_IS_EXACTLY_SM_70,  invoke_count += 1; invoke_count += threadIdx.x;,
    NV_IS_EXACTLY_SM_62,  invoke_count += 1; invoke_count += threadIdx.x;,
    NV_IS_EXACTLY_SM_61,  invoke_count += 1; invoke_count += threadIdx.x;,
    NV_IS_EXACTLY_SM_60, (invoke_count += 1; invoke_count += threadIdx.x;),
    NV_IS_EXACTLY_SM_53, (invoke_count += 1; invoke_count += threadIdx.x;),
    NV_IS_EXACTLY_SM_52, (invoke_count += 1; invoke_count += threadIdx.x;),
    NV_IS_EXACTLY_SM_50, (invoke_count += 1; invoke_count += threadIdx.x;),
    NV_IS_EXACTLY_SM_37, (invoke_count += 1; invoke_count += threadIdx.x;),
    NV_IS_EXACTLY_SM_35, (invoke_count += 1; invoke_count += threadIdx.x;),
    NV_IS_HOST,          (invoke_count += 1;)
  )

  assert(invoke_count == 1);
  invoke_count = 0;

  NV_IF_TARGET(
    NV_IS_HOST,
      invoke_count += 1;,
      invoke_count += 1; invoke_count += threadIdx.x;
  )

  assert(invoke_count == 1);
  invoke_count = 0;

  NV_IF_TARGET(
    NV_IS_DEVICE,
      invoke_count += 1; invoke_count += threadIdx.x;,
      invoke_count += 1;
  )

  assert(invoke_count == 1);
  invoke_count = 0;

  NV_DISPATCH_TARGET(
    NV_NO_TARGET, invoke_count += 0;,
    NV_ANY_TARGET, invoke_count += 1;
  )

  assert(invoke_count == 1);
  invoke_count = 0;

  NV_IF_TARGET(
    NV_IS_HOST,
      printf("Host success\r\n");,
      printf("Device success\r\n");
  )

  NV_DISPATCH_TARGET(
    NV_IS_HOST, (),
    NV_IS_DEVICE, (
      static_assert(NV_TARGET_MINIMUM_SM_INTEGER == (__CUDA_MINIMUM_ARCH__ / 10), "arch mismatch");
      static_assert(nv::target::detail::toint(NV_TARGET_MINIMUM_SM_SELECTOR) == (__CUDA_MINIMUM_ARCH__ / 10), "arch mismatch");
    )
  )
}

#elif defined(TEST_HOST)

void test() {
  constexpr int arch_val = 0;

  // This test ensures that the fallthrough cases are not invoked.
  // SM_80 would imply that SM_72 is available, yet it should not be expanded by the macro
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_80, (static_assert(arch_val == 800, "cuda arch expected 800");),
    NV_PROVIDES_SM_75, (static_assert(arch_val == 750, "cuda arch expected 750");),
    NV_PROVIDES_SM_72, (static_assert(arch_val == 720, "cuda arch expected 720");),
    NV_PROVIDES_SM_70, (static_assert(arch_val == 700, "cuda arch expected 700");),
    NV_PROVIDES_SM_62, (static_assert(arch_val == 620, "cuda arch expected 620");),
    NV_PROVIDES_SM_61, (static_assert(arch_val == 610, "cuda arch expected 610");),
    NV_PROVIDES_SM_60, (static_assert(arch_val == 600, "cuda arch expected 600");),
    NV_PROVIDES_SM_53, (static_assert(arch_val == 530, "cuda arch expected 530");),
    NV_PROVIDES_SM_52, (static_assert(arch_val == 520, "cuda arch expected 520");),
    NV_PROVIDES_SM_50, (static_assert(arch_val == 500, "cuda arch expected 500");),
    NV_PROVIDES_SM_37, (static_assert(arch_val == 370, "cuda arch expected 370");),
    NV_PROVIDES_SM_35, (static_assert(arch_val == 350, "cuda arch expected 350");),
    NV_IS_HOST,        (static_assert(arch_val == 0,   "cuda arch expected 0");)
  )

  // This test is simpler and ensures that only the value matched is invoked, but is roughly the same as the above
  NV_DISPATCH_TARGET(
    NV_IS_EXACTLY_SM_80, (static_assert(arch_val == 800, "cuda arch expected 800");),
    NV_IS_EXACTLY_SM_75, (static_assert(arch_val == 750, "cuda arch expected 750");),
    NV_IS_EXACTLY_SM_72, (static_assert(arch_val == 720, "cuda arch expected 720");),
    NV_IS_EXACTLY_SM_70, (static_assert(arch_val == 700, "cuda arch expected 700");),
    NV_IS_EXACTLY_SM_62, (static_assert(arch_val == 620, "cuda arch expected 620");),
    NV_IS_EXACTLY_SM_61, (static_assert(arch_val == 610, "cuda arch expected 610");),
    NV_IS_EXACTLY_SM_60, (static_assert(arch_val == 600, "cuda arch expected 600");),
    NV_IS_EXACTLY_SM_53, (static_assert(arch_val == 530, "cuda arch expected 530");),
    NV_IS_EXACTLY_SM_52, (static_assert(arch_val == 520, "cuda arch expected 520");),
    NV_IS_EXACTLY_SM_50, (static_assert(arch_val == 500, "cuda arch expected 500");),
    NV_IS_EXACTLY_SM_37, (static_assert(arch_val == 370, "cuda arch expected 370");),
    NV_IS_EXACTLY_SM_35, (static_assert(arch_val == 350, "cuda arch expected 350");),
    NV_IS_HOST,          (static_assert(arch_val == 0,   "cuda arch expected 0");)
  )

  NV_IF_TARGET(
    NV_IS_HOST,
      (static_assert(arch_val == 0, "cuda arch expected 0");),
      (static_assert(arch_val != 0, "cuda arch expected !0");)
  )

  // Some additional tests, but briefly exercise the parenthesis hacks on host compilers
  NV_IF_TARGET(
    NV_IS_DEVICE,
      static_assert(arch_val != 0, "cuda arch expected !0");,
      static_assert(arch_val == 0, "cuda arch expected 0");
  )

  NV_DISPATCH_TARGET(
    NV_IS_DEVICE, static_assert(arch_val != 0, "cuda arch expected !0");,
    NV_IS_HOST,   static_assert(arch_val == 0, "cuda arch expected 0");
  )

  NV_DISPATCH_TARGET(
    NV_NO_TARGET, assert("Should never be hit");,
    NV_ANY_TARGET, static_assert(arch_val == arch_val, "");
  )

  NV_IF_TARGET(
    NV_IS_HOST,
      printf("Host success\r\n");,
      printf("Device success\r\n");
  )

  NV_DISPATCH_TARGET(
    NV_IS_HOST, (),
    NV_IS_DEVICE, (
      static_assert(NV_TARGET_MINIMUM_SM_INTEGER == (__CUDA_ARCH__ / 10), "arch mismatch");
      static_assert(nv::target::detail::toint(NV_TARGET_MINIMUM_SM_SELECTOR) == (__CUDA_ARCH__ / 10), "arch mismatch");
      static_assert(__CUDA_MINIMUM_ARCH__ == __CUDA_ARCH__, "arch mismatch");
    )
  )
}

#endif


int main(int argc, char ** argv)
{
    test();
    return 0;
}
