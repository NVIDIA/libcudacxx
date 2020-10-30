//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits
// XFAIL: apple-clang-6.0
//  The Apple-6 compiler gets is_constructible<void ()> wrong.

// template <class T, class... Args>
//   struct is_constructible;

// MODULES_DEFINES: _LIBCUDACXX_TESTING_FALLBACK_IS_CONSTRUCTIBLE
#define _LIBCUDACXX_TESTING_FALLBACK_IS_CONSTRUCTIBLE
#include <cuda/std/type_traits>
#include "test_macros.h"

#if TEST_STD_VER >= 11 && defined(_LIBCUDACXX_VERSION)
#define LIBCPP11_STATIC_ASSERT(...) static_assert(__VA_ARGS__)
#else
#define LIBCPP11_STATIC_ASSERT(...) ((void)0)
#endif

#pragma diag_suppress declared_but_not_referenced

struct A
{
    __host__ __device__
    explicit A(int);
    __host__ __device__
    A(int, double);
    __host__ __device__
    A(int, long, double);
#if TEST_STD_VER >= 11
private:
#endif
    __host__ __device__
    A(char);
};

struct Base {};
struct Derived : public Base {};

class Abstract
{
    __host__ __device__
    virtual void foo() = 0;
};

class AbstractDestructor
{
    __host__ __device__
    virtual ~AbstractDestructor() = 0;
};

struct PrivateDtor {
    __host__ __device__
  PrivateDtor(int) {}
private:
    __host__ __device__
  ~PrivateDtor() {}
};

struct S {
   template <class T>
    __host__ __device__
#if TEST_STD_VER >= 11
   explicit
#endif
   operator T () const;
};

template <class To>
struct ImplicitTo {
    __host__ __device__
  operator To();
};

#if TEST_STD_VER >= 11
template <class To>
struct ExplicitTo {
    __host__ __device__
   explicit operator To ();
};
#endif


template <class T>
__host__ __device__
void test_is_constructible()
{
    static_assert( (cuda::std::is_constructible<T>::value), "");
#ifndef TEST_COMPILER_C1XX
    // The fallback SFINAE version doesn't work reliable with MSVC, and we don't
    // use it, so waive it.
    LIBCPP11_STATIC_ASSERT((cuda::std::__libcpp_is_constructible<T>::type::value), "");
#endif
#if TEST_STD_VER > 11
    static_assert( cuda::std::is_constructible_v<T>, "");
#endif
}

template <class T, class A0>
__host__ __device__
void test_is_constructible()
{
    static_assert(( cuda::std::is_constructible<T, A0>::value), "");
#ifndef TEST_COMPILER_C1XX
    // The fallback SFINAE version doesn't work reliable with MSVC, and we don't
    // use it, so waive it.
    LIBCPP11_STATIC_ASSERT((cuda::std::__libcpp_is_constructible<T, A0>::type::value), "");
#endif
#if TEST_STD_VER > 11
    static_assert(( cuda::std::is_constructible_v<T, A0>), "");
#endif
}

template <class T, class A0, class A1>
__host__ __device__
void test_is_constructible()
{
    static_assert(( cuda::std::is_constructible<T, A0, A1>::value), "");
#ifndef TEST_COMPILER_C1XX
    // The fallback SFINAE version doesn't work reliable with MSVC, and we don't
    // use it, so waive it.
    LIBCPP11_STATIC_ASSERT((cuda::std::__libcpp_is_constructible<T, A0, A1>::type::value), "");
#endif
#if TEST_STD_VER > 11
    static_assert(( cuda::std::is_constructible_v<T, A0, A1>), "");
#endif
}

template <class T, class A0, class A1, class A2>
__host__ __device__
void test_is_constructible()
{
    static_assert(( cuda::std::is_constructible<T, A0, A1, A2>::value), "");
#ifndef TEST_COMPILER_C1XX
    // The fallback SFINAE version doesn't work reliable with MSVC, and we don't
    // use it, so waive it.
    LIBCPP11_STATIC_ASSERT((cuda::std::__libcpp_is_constructible<T, A0, A1, A2>::type::value), "");
#endif
#if TEST_STD_VER > 11
    static_assert(( cuda::std::is_constructible_v<T, A0, A1, A2>), "");
#endif
}

template <class T>
__host__ __device__
void test_is_not_constructible()
{
    static_assert((!cuda::std::is_constructible<T>::value), "");
#ifndef TEST_COMPILER_C1XX
    // The fallback SFINAE version doesn't work reliable with MSVC, and we don't
    // use it, so waive it.
    LIBCPP11_STATIC_ASSERT((!cuda::std::__libcpp_is_constructible<T>::type::value), "");
#endif
#if TEST_STD_VER > 11
    static_assert((!cuda::std::is_constructible_v<T>), "");
#endif
}

template <class T, class A0>
__host__ __device__
void test_is_not_constructible()
{
    static_assert((!cuda::std::is_constructible<T, A0>::value), "");
#ifndef TEST_COMPILER_C1XX
    // The fallback SFINAE version doesn't work reliable with MSVC, and we don't
    // use it, so waive it.
    LIBCPP11_STATIC_ASSERT((!cuda::std::__libcpp_is_constructible<T, A0>::type::value), "");
#endif
#if TEST_STD_VER > 11
    static_assert((!cuda::std::is_constructible_v<T, A0>), "");
#endif
}

#if defined(TEST_CLANG_VER) && TEST_STD_VER >= 11
template <class T = int, class = decltype(static_cast<T&&>(cuda::std::declval<double&>()))>
__host__ __device__
constexpr bool  clang_disallows_valid_static_cast_test(int) { return false; };

__host__ __device__
constexpr bool clang_disallows_valid_static_cast_test(long) { return true; }

static constexpr bool clang_disallows_valid_static_cast_bug =
    clang_disallows_valid_static_cast_test(0);
#endif


int main(int, char**)
{
    typedef Base B;
    typedef Derived D;

    test_is_constructible<int> ();
    test_is_constructible<int, const int> ();
    test_is_constructible<A, int> ();
    test_is_constructible<A, int, double> ();
    test_is_constructible<A, int, long, double> ();
    test_is_constructible<int&, int&> ();

    test_is_not_constructible<A> ();
#if TEST_STD_VER >= 11
    test_is_not_constructible<A, char> ();
#else
    test_is_constructible<A, char> ();
#endif
    test_is_not_constructible<A, void> ();
    test_is_not_constructible<int, void()>();
    test_is_not_constructible<int, void(&)()>();
    test_is_not_constructible<int, void() const>();
    test_is_not_constructible<int&, void>();
    test_is_not_constructible<int&, void()>();
    test_is_not_constructible<int&, void() const>();
    test_is_not_constructible<int&, void(&)()>();

    test_is_not_constructible<void> ();
    test_is_not_constructible<const void> ();  // LWG 2738
    test_is_not_constructible<volatile void> ();
    test_is_not_constructible<const volatile void> ();
    test_is_not_constructible<int&> ();
    test_is_not_constructible<Abstract> ();
    test_is_not_constructible<AbstractDestructor> ();
    test_is_constructible<int, S>();
    test_is_not_constructible<int&, S>();

    test_is_constructible<void(&)(), void(&)()>();
    test_is_constructible<void(&)(), void()>();
#if TEST_STD_VER >= 11
    test_is_constructible<void(&&)(), void(&&)()>();
    test_is_constructible<void(&&)(), void()>();
    test_is_constructible<void(&&)(), void(&)()>();
#endif

#if TEST_STD_VER >= 11
    test_is_constructible<int const&, int>();
    test_is_constructible<int const&, int&&>();

#ifndef TEST_COMPILER_C1XX
    // This appears to be an MSVC bug, it reproduces with their standard library
    // in 19.20:
    // https://godbolt.org/z/X455LN
    // https://developercommunity.visualstudio.com/content/problem/679848/stdis-constructible-v-incorrectly-returns-false-wi.html
    test_is_constructible<int&&, double&>();
#endif
    test_is_constructible<void(&)(), void(&&)()>();

    test_is_not_constructible<int&, int>();
    test_is_not_constructible<int&, int const&>();
    test_is_not_constructible<int&, int&&>();

    test_is_constructible<int&&, int>();
    test_is_constructible<int&&, int&&>();
    test_is_not_constructible<int&&, int&>();
    test_is_not_constructible<int&&, int const&&>();

    test_is_constructible<Base, Derived>();
    test_is_constructible<Base&, Derived&>();
    test_is_not_constructible<Derived&, Base&>();
    test_is_constructible<Base const&, Derived const&>();
    test_is_not_constructible<Derived const&, Base const&>();
    test_is_not_constructible<Derived const&, Base>();

    test_is_constructible<Base&&, Derived>();
    test_is_constructible<Base&&, Derived&&>();
    test_is_not_constructible<Derived&&, Base&&>();
    test_is_not_constructible<Derived&&, Base>();

    // test that T must also be destructible
    test_is_constructible<PrivateDtor&, PrivateDtor&>();
#if !defined(TEST_COMPILER_C1XX) || 1920 <= _MSC_VER
    test_is_not_constructible<PrivateDtor, int>();
#endif

    test_is_not_constructible<void() const, void() const>();
    test_is_not_constructible<void() const, void*>();

    test_is_constructible<int&, ImplicitTo<int&>>();
    test_is_constructible<const int&, ImplicitTo<int&&>>();
    test_is_constructible<int&&, ImplicitTo<int&&>>();
    test_is_constructible<const int&, ImplicitTo<int>>();

    test_is_not_constructible<B&&, B&>();
    test_is_not_constructible<B&&, D&>();
    test_is_constructible<B&&, ImplicitTo<D&&>>();
    test_is_constructible<B&&, ImplicitTo<D&&>&>();

    test_is_constructible<const int&, ImplicitTo<int&>&>();
    test_is_constructible<const int&, ImplicitTo<int&>>();
    test_is_constructible<const int&, ExplicitTo<int&>&>();
    test_is_constructible<const int&, ExplicitTo<int&>>();

    test_is_constructible<const int&, ExplicitTo<int&>&>();
    test_is_constructible<const int&, ExplicitTo<int&>>();

    // Binding through reference-compatible type is required to perform
    // direct-initialization as described in [over.match.ref] p. 1 b. 1:
    //
    // But the rvalue to lvalue reference binding isn't allowed according to
    // [over.match.ref] despite Clang accepting it.
    test_is_constructible<int&, ExplicitTo<int&>>();

    // This fails almost everywhere.
    //test_is_constructible<const int&, ExplicitTo<int&&>>();

    // TODO add nvbug tracking
#if !defined(TEST_COMPILER_NVCC) && !defined(TEST_COMPILER_NVRTC)
    static_assert(cuda::std::is_constructible<int&&, ExplicitTo<int&&>>::value, "");
#endif

#if defined(TEST_CLANG_VER) && !defined(TEST_COMPILER_NVCC)
    #if TEST_CLANG_VER < 400
        static_assert(clang_disallows_valid_static_cast_bug, "bug still exists");
        // FIXME Clang disallows this construction because it thinks that
        // 'static_cast<int&&>(declval<ExplicitTo<int&&>>())' is ill-formed.
        LIBCPP_STATIC_ASSERT(
            clang_disallows_valid_static_cast_bug !=
            cuda::std::__libcpp_is_constructible<int&&, ExplicitTo<int&&>>::value, "");
        ((void)clang_disallows_valid_static_cast_bug); // Prevent unused warning
    #else
        static_assert(clang_disallows_valid_static_cast_bug == false, "");
        LIBCPP_STATIC_ASSERT(cuda::std::__libcpp_is_constructible<
            int&&, ExplicitTo<int&&>>::value, "");
    #endif
#endif

// FIXME Compilers disagree about the validity of these tests.
#if defined(TEST_CLANG_VER) && !defined(TEST_COMPILER_NVCC)
    test_is_constructible<const int&, ExplicitTo<int>>();
    LIBCPP_STATIC_ASSERT(
        clang_disallows_valid_static_cast_bug !=
        cuda::std::__libcpp_is_constructible<int&&, ExplicitTo<int>>::value, "");
    static_assert(cuda::std::is_constructible<int&&, ExplicitTo<int>>::value, "");
#elif defined(TEST_COMPILER_C1XX) && defined(TEST_COMPILER_NVCC)
    // FIXME NVCC and MSVC disagree about the validity of these tests, and give
    //       different answers in host and device code, which is just wonderful.
#elif defined(TEST_CLANG_VER) && defined(TEST_COMPILER_NVCC)
    // FIXME NVCC fails the assertion below when used with clang.
#elif defined(TEST_COMPILER_NVRTC) || defined(TEST_COMPILER_PGI)
    // FIXME NVRTC also doesn't like these tests.
    // FIXME neither does NVCC+PGI.
#else // GCC and others.
    test_is_not_constructible<const int&, ExplicitTo<int>>();
    test_is_not_constructible<int&&, ExplicitTo<int>>();
#endif

    // Binding through temporary behaves like copy-initialization,
    // see [dcl.init.ref] p. 5, very last sub-bullet:
    test_is_not_constructible<const int&, ExplicitTo<double&&>>();
    test_is_not_constructible<int&&, ExplicitTo<double&&>>();

// TODO: Remove this workaround once Clang <= 3.7 are no longer used regularly.
// In those compiler versions the __is_constructible builtin gives the wrong
// results for abominable function types.
#if (defined(TEST_APPLE_CLANG_VER) && TEST_APPLE_CLANG_VER < 703) \
 || (defined(TEST_CLANG_VER) && TEST_CLANG_VER < 308)
#define WORKAROUND_CLANG_BUG
#endif
#if !defined(WORKAROUND_CLANG_BUG)
    test_is_not_constructible<void()>();
    test_is_not_constructible<void() const> ();
    test_is_not_constructible<void() volatile> ();
    test_is_not_constructible<void() &> ();
    test_is_not_constructible<void() &&> ();
#endif
#endif // TEST_STD_VER >= 11

  return 0;
}
