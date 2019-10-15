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

#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"


struct R { };
struct A1 { };
struct A2 { };
struct A3 { };

#define DECLARE_FUNCTIONS_WITH_QUALS(N, ...)                              \
  struct f0_##N  { R operator()() __VA_ARGS__           { return {}; } }; \
  struct f1_##N  { R operator()(A1) __VA_ARGS__         { return {}; } }; \
  struct f2_##N  { R operator()(A1, A2) __VA_ARGS__     { return {}; } }; \
  struct f3_##N  { R operator()(A1, A2, A3) __VA_ARGS__ { return {}; } }  \
/**/

DECLARE_FUNCTIONS_WITH_QUALS(0, /* nothing */);
DECLARE_FUNCTIONS_WITH_QUALS(1, const);
DECLARE_FUNCTIONS_WITH_QUALS(2, volatile);
DECLARE_FUNCTIONS_WITH_QUALS(3, const volatile);
DECLARE_FUNCTIONS_WITH_QUALS(4, &);
DECLARE_FUNCTIONS_WITH_QUALS(5 , const &);
DECLARE_FUNCTIONS_WITH_QUALS(6 , volatile &);
DECLARE_FUNCTIONS_WITH_QUALS(7 , const volatile &);
DECLARE_FUNCTIONS_WITH_QUALS(8 , noexcept);
DECLARE_FUNCTIONS_WITH_QUALS(9 , const noexcept);
DECLARE_FUNCTIONS_WITH_QUALS(10, volatile noexcept);
DECLARE_FUNCTIONS_WITH_QUALS(11, const volatile noexcept);
DECLARE_FUNCTIONS_WITH_QUALS(12, & noexcept);
DECLARE_FUNCTIONS_WITH_QUALS(13, const & noexcept);
DECLARE_FUNCTIONS_WITH_QUALS(14, volatile & noexcept);
DECLARE_FUNCTIONS_WITH_QUALS(15, const volatile & noexcept);

int main() {
#define CHECK_FUNCTIONS(N)                                                    \
  do {                                                                        \
    /* implicit */                                                            \
    cuda::std::function g0 = f0_##N{};                                              \
    ASSERT_SAME_TYPE(decltype(g0), cuda::std::function<R()>);                       \
                                                                              \
    cuda::std::function g1 = f1_##N{};                                              \
    ASSERT_SAME_TYPE(decltype(g1), cuda::std::function<R(A1)>);                     \
                                                                              \
    cuda::std::function g2 = f2_##N{};                                              \
    ASSERT_SAME_TYPE(decltype(g2), cuda::std::function<R(A1, A2)>);                 \
                                                                              \
    cuda::std::function g3 = f3_##N{};                                              \
    ASSERT_SAME_TYPE(decltype(g3), cuda::std::function<R(A1, A2, A3)>);             \
                                                                              \
    /* explicit */                                                            \
    cuda::std::function g4{f0_##N{}};                                               \
    ASSERT_SAME_TYPE(decltype(g4), cuda::std::function<R()>);                       \
                                                                              \
    cuda::std::function g5{f1_##N{}};                                               \
    ASSERT_SAME_TYPE(decltype(g5), cuda::std::function<R(A1)>);                     \
                                                                              \
    cuda::std::function g6{f2_##N{}};                                               \
    ASSERT_SAME_TYPE(decltype(g6), cuda::std::function<R(A1, A2)>);                 \
                                                                              \
    cuda::std::function g7{f3_##N{}};                                               \
    ASSERT_SAME_TYPE(decltype(g7), cuda::std::function<R(A1, A2, A3)>);             \
                                                                              \
    /* from cuda::std::function */                                                  \
    cuda::std::function<R(A1)> unary;                                               \
    cuda::std::function g8 = unary;                                                 \
    ASSERT_SAME_TYPE(decltype(g8), cuda::std::function<R(A1)>);                     \
                                                                              \
    cuda::std::function g9 = cuda::std::move(unary);                                      \
    ASSERT_SAME_TYPE(decltype(g9), cuda::std::function<R(A1)>);                     \
                                                                              \
    cuda::std::function<R(A1&&)> unary_ref;                                         \
    cuda::std::function g10 = unary_ref;                                            \
    ASSERT_SAME_TYPE(decltype(g10), cuda::std::function<R(A1&&)>);                  \
                                                                              \
    cuda::std::function g11 = cuda::std::move(unary_ref);                                 \
    ASSERT_SAME_TYPE(decltype(g11), cuda::std::function<R(A1&&)>);                  \
  } while (false)                                                             \
/**/

  // Make sure we can deduce from function objects with valid call operators
  CHECK_FUNCTIONS(0);
  CHECK_FUNCTIONS(1);
  CHECK_FUNCTIONS(2);
  CHECK_FUNCTIONS(3);
  CHECK_FUNCTIONS(4);
  CHECK_FUNCTIONS(5);
  CHECK_FUNCTIONS(6);
  CHECK_FUNCTIONS(7);
  CHECK_FUNCTIONS(8);
  CHECK_FUNCTIONS(9);
  CHECK_FUNCTIONS(10);
  CHECK_FUNCTIONS(11);
  CHECK_FUNCTIONS(12);
  CHECK_FUNCTIONS(13);
  CHECK_FUNCTIONS(14);
  CHECK_FUNCTIONS(15);
}

// Make sure we fail in a SFINAE-friendly manner when we try to deduce
// from a type without a valid call operator.
template <typename F, typename = decltype(cuda::std::function{cuda::std::declval<F>()})>
constexpr bool can_deduce() { return true; }
template <typename F>
constexpr bool can_deduce(...) { return false; }

struct invalid1 { };
struct invalid2 {
  template <typename ...Args>
  void operator()(Args ...);
};
struct invalid3 {
  void operator()(int);
  void operator()(long);
};
static_assert(!can_deduce<invalid1>());
static_assert(!can_deduce<invalid2>());
static_assert(!can_deduce<invalid3>());
