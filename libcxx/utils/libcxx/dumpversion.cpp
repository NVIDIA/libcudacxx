//===----------------------------------------------------------------------===##
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===////

#include <stdio.h>

int main()
{
  char const* compiler_type = "unknown";
  unsigned major_version = 0;
  unsigned minor_version = 0;
  unsigned patch_level   = 0;

  #if defined(__NVCC__)
    compiler_type = "nvcc";
    major_version = __CUDACC_VER_MAJOR__;
    minor_version = __CUDACC_VER_MINOR__;
    patch_level   = __CUDACC_VER_BUILD__;
  #elif defined(__clang__)
    // Treat apple's llvm fork differently.
    #if defined(__apple_build_version__)
      compiler_type = "apple-clang";
    #else
      compiler_type = "clang";
    #endif
    major_version = __clang_major__;
    minor_version = __clang_minor__;
    patch_level   = __clang_patchlevel__;
  #elif defined(_MSC_VER)
    compiler_type = "msvc";
    major_version = _MSC_FULL_VER / 10000000;
    minor_version = _MSC_FULL_VER / 100000 % 100;
    patch_level   = _MSC_FULL_VER % 100000;
  #elif defined(__GNUC__)
    compiler_type = "gcc";
    major_version = __GNUC__;
    minor_version = __GNUC_MINOR__;
    patch_level   = __GNUC_PATCHLEVEL__;
  #endif

  printf("(\"%s\", (%d, %d, %d))\n", 
         compiler_type, major_version, minor_version, patch_level);
}

