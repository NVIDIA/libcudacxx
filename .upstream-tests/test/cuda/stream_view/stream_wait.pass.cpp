//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/stream_view>
#include <cassert>

int main(int argc, char** argv){

#ifndef __CUDA_ARCH__
  cudaStream_t s;
  cudaStreamCreate(&s);
  cuda::stream_view sv{s};
  try{
    sv.wait();
  } catch(...) {
      assert(false && "Should not have thrown");
  }
#endif

  return 0;
}