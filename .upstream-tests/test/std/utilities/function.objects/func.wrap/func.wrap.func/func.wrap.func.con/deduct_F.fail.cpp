//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// template<class F>
// function(F) -> function<see-below>;

// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: libcpp-no-deduction-guides

// The deduction guides for cuda::std::function do not handle rvalue-ref qualified
// call operators and C-style variadics. It also doesn't deduce from nullptr_t.
// Make sure we stick to the specification.

#include <cuda/std/functional>
#include <cuda/std/type_traits>


struct R { };
struct f0 { R operator()() && { return {}; } };
struct f1 { R operator()(int, ...) { return {}; } };

int main() {
    cuda::std::function f = f0{}; // expected-error{{no viable constructor or deduction guide for deduction of template arguments of 'function'}}
    cuda::std::function g = f1{}; // expected-error{{no viable constructor or deduction guide for deduction of template arguments of 'function'}}
    cuda::std::function h = nullptr; // expected-error{{no viable constructor or deduction guide for deduction of template arguments of 'function'}}
}
