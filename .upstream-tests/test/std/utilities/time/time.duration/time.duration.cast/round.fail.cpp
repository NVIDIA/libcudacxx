//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <cuda/std/chrono>

// round

// template <class ToDuration, class Rep, class Period>
//   ToDuration
//   round(const duration<Rep, Period>& d);

// ToDuration shall be an instantiation of duration.

// .fail. expects compilation to fail, but this would only fail at runtime with NVRTC
// UNSUPPORTED: nvrtc

#include <cuda/std/chrono>

int main(int, char**)
{
    cuda::std::chrono::round<int>(cuda::std::chrono::milliseconds(3));

  return 0;
}
