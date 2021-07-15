//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cuda/memory_resource>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/stream_view>
#include <memory>
#include <tuple>
#include <vector>

struct dummy_context {};

int main(int argc, char **argv) {
#ifndef __CUDA_ARCH__
  {
    cuda::resource_view<cuda::memory_resource<cuda::memory_kind::host, cuda::any_context>,
                        cuda::memory_access::host> v_any;
    cuda::resource_view<cuda::memory_resource<cuda::memory_kind::host, dummy_context>,
                        cuda::memory_access::host> v_dummy;
    v_any = v_dummy;
  }
#endif
  return 0;
}
