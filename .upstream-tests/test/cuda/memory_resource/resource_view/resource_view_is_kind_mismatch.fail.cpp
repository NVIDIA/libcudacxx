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

int main(int argc, char **argv) {
#ifndef __CUDA_ARCH__
  cuda::resource_view<cuda::is_kind<cuda::memory_kind::managed> managed;
  cuda::resource_view<cuda::is_kind<cuda::memory_kind::host>> host;

  // Despite managed having a superset of properties of host,
  // this should fail because the kind is a property in itself.
  host = managed;
#endif
  return 0;
}
