//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// reference_wrapper

// template <ObjectType T> reference_wrapper<T> ref(reference_wrapper<T>t);

#include <cuda/std/functional>
#include <cuda/std/cassert>

#include "counting_predicates.h"

__host__ __device__
bool is5 ( int i ) { return i == 5; }

template <typename T>
__host__ __device__
bool call_pred ( T pred ) { return pred(5); }

int main(int, char**)
{
    {
    int i = 0;
    cuda::std::reference_wrapper<int> r1 = cuda::std::ref(i);
    cuda::std::reference_wrapper<int> r2 = cuda::std::ref(r1);
    assert(&r2.get() == &i);
    }
    {
    unary_counting_predicate<bool(*)(int), int> cp(is5);
    assert(!cp(6));
    assert(cp.count() == 1);
    assert(call_pred(cp));
    assert(cp.count() == 1);
    assert(call_pred(cuda::std::ref(cp)));
    assert(cp.count() == 2);
    }

  return 0;
}
