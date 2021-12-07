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
  cuda::pmr::resource_ptr<cuda::pmr::memory_access::host, cuda::pmr::oversubscribable, cuda::pmr::memory_location::host> props_only;
  cuda::pmr::resource_ptr<cuda::pmr::of_kind<d<cuda::pmr::memory_kind::host>> kind_only;
  kind_only = props_only;  // no conversion from a list of properties back to memory kind
#endif
  return 0;
}
