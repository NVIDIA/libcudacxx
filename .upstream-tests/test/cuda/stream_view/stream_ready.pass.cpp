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
#include "../exception_helper.h"

int main(int argc, char** argv){

#ifndef __CUDA_ARCH__
  cudaStream_t s;
  cudaStreamCreate(&s);
  cuda::stream_view sv{s};
  _LIBCUDACXX_TEST_TRY {
    assert(sv.ready());
  } _LIBCUDACXX_TEST_CATCH(...) {
    assert(false && "Should not have thrown");
  }
  cudaStreamDestroy(s);
#endif

  return 0;
}
