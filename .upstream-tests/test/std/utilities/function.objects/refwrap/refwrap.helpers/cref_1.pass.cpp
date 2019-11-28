//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// reference_wrapper

// template <ObjectType T> reference_wrapper<const T> cref(const T& t);

#include <cuda/std/functional>
#include <cuda/std/cassert>

int main(int, char**)
{
    int i = 0;
    cuda::std::reference_wrapper<const int> r = cuda::std::cref(i);
    assert(&r.get() == &i);

  return 0;
}
