//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This test serves as a canary for when this issue is fixed in NVCC
//
// UNSUPPORTED: windows, icc, pgi
// XFAIL: clang-9 && c++11
// XFAIL: clang-10 && c++11
// gcc 10.0 is expected to pass, but later versions do not.
// XFAIL: gcc-10 && c++11


#if defined(__has_builtin)
#if __has_builtin(__builtin_is_constant_evaluated)
#define BUILTIN_CONSTANT_EVAL() __builtin_is_constant_evaluated()
#endif
#endif

#ifndef BUILTIN_CONSTANT_EVAL
#define BUILTIN_CONSTANT_EVAL() true
#endif

__device__ __host__ inline constexpr bool constant_eval() {
  return BUILTIN_CONSTANT_EVAL();
}

int main(int, char**) {
  static_assert(constant_eval(), "");
  return 0;
}
