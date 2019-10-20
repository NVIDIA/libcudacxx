//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/functional>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char **)
{
    assert(cuda::std::minus<int>()(2, 1) == 1);
#if __cplusplus >= 201402LL
    assert(cuda::std::minus<>()(2, 1) == 1);
#endif

    return 0;
}

