//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// template<class R, class ...Args>
// function(R(*)(Args...)) -> function<R(Args...)>;

// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: libcpp-no-deduction-guides

#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include "test_macros.h"


struct R { };
struct A1 { };
struct A2 { };
struct A3 { };

R f0() { return {}; }
R f1(A1) { return {}; }
R f2(A1, A2) { return {}; }
R f3(A1, A2, A3) { return {}; }
R f4(A1 = {}) { return {}; }

int main() {
  {
    // implicit
    cuda::std::function a = f0;
    ASSERT_SAME_TYPE(decltype(a), cuda::std::function<R()>);

    cuda::std::function b = &f0;
    ASSERT_SAME_TYPE(decltype(b), cuda::std::function<R()>);

    // explicit
    cuda::std::function c{f0};
    ASSERT_SAME_TYPE(decltype(c), cuda::std::function<R()>);

    cuda::std::function d{&f0};
    ASSERT_SAME_TYPE(decltype(d), cuda::std::function<R()>);
  }
  {
    // implicit
    cuda::std::function a = f1;
    ASSERT_SAME_TYPE(decltype(a), cuda::std::function<R(A1)>);

    cuda::std::function b = &f1;
    ASSERT_SAME_TYPE(decltype(b), cuda::std::function<R(A1)>);

    // explicit
    cuda::std::function c{f1};
    ASSERT_SAME_TYPE(decltype(c), cuda::std::function<R(A1)>);

    cuda::std::function d{&f1};
    ASSERT_SAME_TYPE(decltype(d), cuda::std::function<R(A1)>);
  }
  {
    // implicit
    cuda::std::function a = f2;
    ASSERT_SAME_TYPE(decltype(a), cuda::std::function<R(A1, A2)>);

    cuda::std::function b = &f2;
    ASSERT_SAME_TYPE(decltype(b), cuda::std::function<R(A1, A2)>);

    // explicit
    cuda::std::function c{f2};
    ASSERT_SAME_TYPE(decltype(c), cuda::std::function<R(A1, A2)>);

    cuda::std::function d{&f2};
    ASSERT_SAME_TYPE(decltype(d), cuda::std::function<R(A1, A2)>);
  }
  {
    // implicit
    cuda::std::function a = f3;
    ASSERT_SAME_TYPE(decltype(a), cuda::std::function<R(A1, A2, A3)>);

    cuda::std::function b = &f3;
    ASSERT_SAME_TYPE(decltype(b), cuda::std::function<R(A1, A2, A3)>);

    // explicit
    cuda::std::function c{f3};
    ASSERT_SAME_TYPE(decltype(c), cuda::std::function<R(A1, A2, A3)>);

    cuda::std::function d{&f3};
    ASSERT_SAME_TYPE(decltype(d), cuda::std::function<R(A1, A2, A3)>);
  }
  // Make sure defaulted arguments don't mess up the deduction
  {
    // implicit
    cuda::std::function a = f4;
    ASSERT_SAME_TYPE(decltype(a), cuda::std::function<R(A1)>);

    cuda::std::function b = &f4;
    ASSERT_SAME_TYPE(decltype(b), cuda::std::function<R(A1)>);

    // explicit
    cuda::std::function c{f4};
    ASSERT_SAME_TYPE(decltype(c), cuda::std::function<R(A1)>);

    cuda::std::function d{&f4};
    ASSERT_SAME_TYPE(decltype(d), cuda::std::function<R(A1)>);
  }
}
