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
#include <type_traits>

int main(int argc, char** argv){

#ifndef __CUDA_ARCH__
  static_assert(
      std::is_same<cuda::stream_view::value_type, cudaStream_t>::value, "");
  cuda::stream_view s;
  assert(s.get() == cudaStream_t{0});
#endif

  return 0;
}
