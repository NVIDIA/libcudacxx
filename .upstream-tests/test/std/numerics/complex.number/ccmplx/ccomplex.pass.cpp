//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

// <cuda/std/ccomplex>

#include <cuda/std/ccomplex>

#include "test_macros.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

int main(int, char**)
{
    cuda::std::complex<double> d;
    unused(d);

  return 0;
}
