//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70

#include "pipeline_memcpy_async_thread_scope_generic.h"

int main(int argc, char ** argv)
{
    test_select_source<cuda::thread_scope_block, int8_t>();

    return 0;
}
