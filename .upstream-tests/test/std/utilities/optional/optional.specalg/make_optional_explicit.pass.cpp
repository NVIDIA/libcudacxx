//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <cuda/std/optional>

// template <class T, class... Args>
//   constexpr optional<T> make_optional(Args&&... args);

#include <cuda/std/optional>
// #include <cuda/std/string>
// #include <cuda/std/memory>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        constexpr auto opt = cuda::std::make_optional<int>('a');
        static_assert(*opt == int('a'), "");
        assert(*opt == int('a'));
    }
    /*
    {
        cuda::std::string s = "123";
        auto opt = cuda::std::make_optional<cuda::std::string>(s);
        assert(*opt == "123");
    }
    {
        cuda::std::unique_ptr<int> s = cuda::std::make_unique<int>(3);
        auto opt = cuda::std::make_optional<cuda::std::unique_ptr<int>>(cuda::std::move(s));
        assert(**opt == 3);
        assert(s == nullptr);
    }
    {
        auto opt = cuda::std::make_optional<cuda::std::string>(4u, 'X');
        assert(*opt == "XXXX");
    }
    */

  return 0;
}
