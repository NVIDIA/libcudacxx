//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// Throwing bad_optional_access is supported starting in macosx10.13
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}} && !no-exceptions

// <cuda/std/optional>
//
// template <class T>
//   constexpr optional<decay_t<T>> make_optional(T&& v);

#include <cuda/std/optional>
// #include <cuda/std/string>
// #include <cuda/std/memory>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        int arr[10];
        auto opt = cuda::std::make_optional(arr);
        ASSERT_SAME_TYPE(decltype(opt), cuda::std::optional<int*>);
        assert(*opt == arr);
    }
    {
        constexpr auto opt = cuda::std::make_optional(2);
        ASSERT_SAME_TYPE(decltype(opt), const cuda::std::optional<int>);
        static_assert(opt.value() == 2, "");
    }
    {
        auto opt = cuda::std::make_optional(2);
        ASSERT_SAME_TYPE(decltype(opt), cuda::std::optional<int>);
        assert(*opt == 2);
    }
    /*
    {
        const cuda::std::string s = "123";
        auto opt = cuda::std::make_optional(s);
        ASSERT_SAME_TYPE(decltype(opt), cuda::std::optional<cuda::std::string>);
        assert(*opt == "123");
    }
    {
        cuda::std::unique_ptr<int> s = cuda::std::make_unique<int>(3);
        auto opt = cuda::std::make_optional(cuda::std::move(s));
        ASSERT_SAME_TYPE(decltype(opt), cuda::std::optional<cuda::std::unique_ptr<int>>);
        assert(**opt == 3);
        assert(s == nullptr);
    }
    */

  return 0;
}
