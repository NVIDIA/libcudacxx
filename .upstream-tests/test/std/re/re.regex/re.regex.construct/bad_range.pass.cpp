//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-no-exceptions
// <cuda/std/regex>

// template <class charT, class traits = regex_traits<charT>> class basic_regex;

// template <class ST, class SA>
//    basic_regex(const basic_string<charT, ST, SA>& s);

#include <cuda/std/regex>
#include <cuda/std/cassert>
#include "test_macros.h"

static bool error_range_thrown(const char *pat)
{
    bool result = false;
    try {
        cuda::std::regex re(pat);
    } catch (const cuda::std::regex_error &ex) {
        result = (ex.code() == cuda::std::regex_constants::error_range);
    }
    return result;
}

int main(int, char**)
{
    assert(error_range_thrown("([\\w-a])"));
    assert(error_range_thrown("([a-\\w])"));

  return 0;
}
