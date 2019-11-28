//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// reference_wrapper

// template <ObjectType T> reference_wrapper<const T> cref(reference_wrapper<T> t);

#include <cuda/std/functional>
#include <cuda/std/cassert>

int main(int, char**)
{
    const int i = 0;
    cuda::std::reference_wrapper<const int> r1 = cuda::std::cref(i);
    cuda::std::reference_wrapper<const int> r2 = cuda::std::cref(r1);
    assert(&r2.get() == &i);

  return 0;
}
