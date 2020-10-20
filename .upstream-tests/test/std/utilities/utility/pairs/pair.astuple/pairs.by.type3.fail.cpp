//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// UNSUPPORTED: nvrtc

#include <cuda/std/utility>
#include <cuda/std/complex>

#include <cuda/std/cassert>

int main(int, char**)
{
    // cuda/std/memory not supported
    /*
    typedef cuda::std::unique_ptr<int> upint;
    cuda::std::pair<upint, int> t(upint(new int(4)), 23);
    upint p = cuda::std::get<upint>(t);
    */

  return 0;
}
