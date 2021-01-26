//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/stream>
#include <cassert>

int main(int argc, char** argv){

#ifndef __CUDA_ARCH__
  cudaStream_t s = reinterpret_cast<cudaStream_t>(42);
  cuda::stream_view sv0{s};
  cuda::stream_view sv1{s};
  cuda::stream_view sv2{};
  assert(sv0 == sv0);
  assert(sv0 == sv1);
  assert(sv0 != sv2);
#endif

  return 0;
}
