//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/iostream>
#include <cuda/std/cassert>

#include "test_macros.h"

// Test to make sure that the streams only get initialized once
// Taken from https://bugs.llvm.org/show_bug.cgi?id=43300

// The dylibs shipped on macOS so far do not contain the fix for PR43300, so
// this test fails.
// XFAIL: with_system_cxx_lib=macosx10.15
// XFAIL: with_system_cxx_lib=macosx10.14
// XFAIL: with_system_cxx_lib=macosx10.13
// XFAIL: with_system_cxx_lib=macosx10.12
// XFAIL: with_system_cxx_lib=macosx10.11
// XFAIL: with_system_cxx_lib=macosx10.10
// XFAIL: with_system_cxx_lib=macosx10.9
// XFAIL: with_system_cxx_lib=macosx10.8
// XFAIL: with_system_cxx_lib=macosx10.7

int main(int, char**)
{

    cuda::std::cout << "Hello!";
    cuda::std::ios_base::fmtflags stock_flags = cuda::std::cout.flags();

    cuda::std::cout << cuda::std::boolalpha << true;
    cuda::std::ios_base::fmtflags ba_flags = cuda::std::cout.flags();
    assert(stock_flags != ba_flags);

    cuda::std::ios_base::Init init_streams;
    cuda::std::ios_base::fmtflags after_init = cuda::std::cout.flags();
    assert(after_init == ba_flags);

    return 0;
}
