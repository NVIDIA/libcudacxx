//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: pre-sm-60

// .fail. expects compilation to fail, but this would only fail at runtime with NVRTC
// UNSUPPORTED: nvrtc

// <atomic>

// struct atomic_flag

// atomic_flag& operator=(const atomic_flag&) = delete;

#include <atomic>
#include <cassert>

int main(int, char**)
{
    std::atomic_flag f0;
    volatile std::atomic_flag f;
    f = f0;

  return 0;
}
