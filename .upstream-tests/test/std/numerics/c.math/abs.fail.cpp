//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cmath>

#include "test_macros.h"

int main(int, char**)
{
    unsigned int ui = -5;
    ui = cuda::std::abs(ui); // expected-error {{call to 'abs' is ambiguous}}

    unsigned char uc = -5;
    uc = cuda::std::abs(uc); // expected-error {{taking the absolute value of unsigned type 'unsigned char' has no effect}}

    unsigned short us = -5;
    us = cuda::std::abs(us); // expected-error {{taking the absolute value of unsigned type 'unsigned short' has no effect}}

    unsigned long ul = -5;
    ul = cuda::std::abs(ul); // expected-error {{call to 'abs' is ambiguous}}

    unsigned long long ull = -5;
    ull = ::abs(ull); // expected-error {{call to 'abs' is ambiguous}}

    return 0;
}

