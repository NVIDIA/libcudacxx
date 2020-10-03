//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: gcc-4

// <utility>

// template <class T1, class T2> struct pair

// explicit(see-below) constexpr pair();

// NOTE: The SFINAE on the default constructor is tested in
//       default-sfinae.pass.cpp


#include <cuda/std/utility>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "archetypes.h"

int main(int, char**)
{
    {
        typedef cuda::std::pair<float, short*> P;
        P p;
        assert(p.first == 0.0f);
        assert(p.second == nullptr);
    }
#if TEST_STD_VER >= 11
    {
        typedef cuda::std::pair<float, short*> P;
        constexpr P p;
        static_assert(p.first == 0.0f, "");
        static_assert(p.second == nullptr, "");
    }
    {
        using NoDefault = ImplicitTypes::NoDefault;
        using P = cuda::std::pair<int, NoDefault>;
        static_assert(!cuda::std::is_default_constructible<P>::value, "");
        using P2 = cuda::std::pair<NoDefault, int>;
        static_assert(!cuda::std::is_default_constructible<P2>::value, "");
    }
    {
        struct Base { };
        struct Derived : Base { protected: Derived() = default; };
        static_assert(!cuda::std::is_default_constructible<cuda::std::pair<Derived, int> >::value, "");
    }
#endif

  return 0;
}
