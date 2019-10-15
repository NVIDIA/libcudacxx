//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/string>

// template<class charT, class traits, class Allocator>
//   basic_ostream<charT, traits>&
//   operator<<(basic_ostream<charT, traits>& os,
//              const basic_string_view<charT,traits> str);

#include <cuda/std/string_view>
#include <cuda/std/iosfwd>

template <class SV, class = void>
struct HasDecl : cuda::std::false_type {};
template <class SV>
struct HasDecl<SV, decltype(static_cast<void>(cuda::std::declval<cuda::std::ostream&>() << cuda::std::declval<SV&>()))> : cuda::std::true_type {};

int main() {
  static_assert(HasDecl<cuda::std::string_view>::value, "streaming operator declaration not present");
}
