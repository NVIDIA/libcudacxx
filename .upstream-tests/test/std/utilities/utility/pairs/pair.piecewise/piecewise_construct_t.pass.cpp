//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <utility>

// struct piecewise_construct_t { explicit piecewise_construct_t() = default; };
// constexpr piecewise_construct_t piecewise_construct = piecewise_construct_t();

#include <cuda/std/utility>

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

int main(int, char**) {
    cuda::std::piecewise_construct_t x = cuda::std::piecewise_construct;
    unused(x);

    return 0;
}
