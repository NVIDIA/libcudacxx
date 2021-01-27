//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/cudaexcept>
#include <cuda/std/type_traits>

#include <cassert>
#include <iostream>

#include <cuda_runtime_api.h>

void test_throw() { throw cuda::cuda_error{cudaErrorInvalidKernelImage}; }

int main(int argc, char **argv) {

#ifndef __CUDA_ARCH__
  static_assert(std::is_base_of<::std::runtime_error, cuda::cuda_error>::value,
                "");

  cuda::cuda_error e{cudaErrorMemoryAllocation};
  assert(e.what() == std::string{"cudaErrorMemoryAllocation: out of memory"});
  assert(e.code() == cudaErrorMemoryAllocation);

  std::string msg = "test message";
  cuda::cuda_error e2{cudaErrorMemoryAllocation, msg};
  assert(e2.what() == msg + ": cudaErrorMemoryAllocation: out of memory");
  assert(e2.code() == cudaErrorMemoryAllocation);

  try {
    test_throw();
  } catch (cuda::cuda_error const &e) {
    assert(e.what() ==
           std::string{
               "cudaErrorInvalidKernelImage: device kernel image is invalid"});
    assert(e.code() == cudaErrorInvalidKernelImage);
  }

  try {
    test_throw();
  } catch (::std::runtime_error const &e) {
    assert(e.what() ==
           std::string{
               "cudaErrorInvalidKernelImage: device kernel image is invalid"});
  }
#endif // __CUDA_ARCH__

  return 0;
}
