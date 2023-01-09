// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_SIZE_H
#define _LIBCUDACXX___RANGES_SIZE_H

#ifndef __cuda_std__
#include <__config>
#include <cstddef>
#endif // __cuda_std__

#include "../__concepts/arithmetic.h"
#include "../__concepts/class_or_enum.h"
#include "../__iterator/concepts.h"
#include "../__iterator/iterator_traits.h"
#include "../__memory/pointer_traits.h"
#include "../__ranges/access.h"
#include "../__type_traits/make_unsigned.h"
#include "../__type_traits/make_signed.h"
#include "../__type_traits/remove_cvref.h"
#include "../__utility/auto_cast.h"
#include "../__utility/declval.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _LIBCUDACXX_STD_VER > 14

  template<class>
  _LIBCUDACXX_INLINE_VAR constexpr bool disable_sized_range = false;

// [range.prim.size]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__size)
  template<class _Tp>
  void size(_Tp&) = delete;
  template<class _Tp>
  void size(const _Tp&) = delete;

  template <class _Tp>
  _LIBCUDACXX_CONCEPT __size_enabled = !disable_sized_range<remove_cvref_t<_Tp>>;

#if _LIBCUDACXX_STD_VER > 17
  template <class _Tp>
  concept __member_size =
    __size_enabled<_Tp> &&
    __workaround_52970<_Tp> &&
    requires(_Tp&& __t) {
      { _LIBCUDACXX_AUTO_CAST(__t.size()) } -> __integer_like;
    };

  template <class _Tp>
  concept __unqualified_size =
    __size_enabled<_Tp> &&
    !__member_size<_Tp> &&
    __class_or_enum<remove_cvref_t<_Tp>> &&
    requires(_Tp&& __t) {
      { _LIBCUDACXX_AUTO_CAST(size(__t)) } -> __integer_like;
    };

  template <class _Tp>
  concept __difference =
    !__member_size<_Tp> &&
    !__unqualified_size<_Tp> &&
    __class_or_enum<remove_cvref_t<_Tp>> &&
    requires(_Tp&& __t) {
      { ranges::begin(__t) } -> forward_iterator;
      { ranges::end(__t) } -> sized_sentinel_for<decltype(ranges::begin(_CUDA_VSTD::declval<_Tp>()))>;
    };
#else // ^^^ CXX20 ^^^ / vvv CXX17 vvv
  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __member_size_,
    requires(_Tp&& __t)(
      requires(__size_enabled<_Tp>),
      requires(__workaround_52970<_Tp>),
      requires(__integer_like<decltype(_LIBCUDACXX_AUTO_CAST(__t.size()))>)
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __member_size = _LIBCUDACXX_FRAGMENT(__member_size_, _Tp);

  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __unqualified_size_,
    requires(_Tp&& __t)(
      requires(__size_enabled<_Tp>),
      requires(!__member_size<_Tp>),
      requires(__class_or_enum<remove_cvref_t<_Tp>>),
      requires(__integer_like<decltype(_LIBCUDACXX_AUTO_CAST(size(__t)))>)
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __unqualified_size = _LIBCUDACXX_FRAGMENT(__unqualified_size_, _Tp);
  
  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __difference_,
    requires(_Tp&& __t)(
      requires(!__member_size<_Tp>),
      requires(!__unqualified_size<_Tp>),
      requires(__class_or_enum<remove_cvref_t<_Tp>>),
      requires(forward_iterator<decltype(ranges::begin(__t))>),
      requires(sized_sentinel_for<decltype(ranges::end(__t)), decltype(ranges::begin(_CUDA_VSTD::declval<_Tp>()))>)
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __difference = _LIBCUDACXX_FRAGMENT(__difference_, _Tp);
#endif // _LIBCUDACXX_STD_VER < 20

  struct __fn {
    // `[range.prim.size]`: the array case (for rvalues).
    template <class _Tp, size_t _Sz>
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY 
    constexpr size_t operator()(_Tp (&&)[_Sz]) const noexcept {
      return _Sz;
    }

    // `[range.prim.size]`: the array case (for lvalues).
    template <class _Tp, size_t _Sz>
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY 
    constexpr size_t operator()(_Tp (&)[_Sz]) const noexcept {
      return _Sz;
    }

    // `[range.prim.size]`: `auto(t.size())` is a valid expression.
  _LIBCUDACXX_TEMPLATE(class _Tp)
    (requires __member_size<_Tp>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY 
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(__t.size()))) {
      return _LIBCUDACXX_AUTO_CAST(__t.size());
    }

    // `[range.prim.size]`: `auto(size(t))` is a valid expression.
  _LIBCUDACXX_TEMPLATE(class _Tp)
    (requires __unqualified_size<_Tp>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY 
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(size(__t)))) {
      return _LIBCUDACXX_AUTO_CAST(size(__t));
    }

    // [range.prim.size]: the `to-unsigned-like` case.
  _LIBCUDACXX_TEMPLATE(class _Tp)
    (requires __difference<_Tp>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY 
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(_CUDA_VSTD::__to_unsigned_like(ranges::end(__t) - ranges::begin(__t))))
      -> decltype(      _CUDA_VSTD::__to_unsigned_like(ranges::end(__t) - ranges::begin(__t)))
      { return          _CUDA_VSTD::__to_unsigned_like(ranges::end(__t) - ranges::begin(__t));
    }
  };
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
   _LIBCUDACXX_CPO_ACCESSIBILITY _LIBCUDACXX_INLINE_VAR constexpr auto size = __size::__fn{};
} // namespace __cpo

// [range.prim.ssize]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__ssize)
#if _LIBCUDACXX_STD_VER > 17
  template<class _Tp>
  concept __can_ssize_ptrdiff = requires (_Tp&& __t) {
    typename make_signed_t<decltype(ranges::size(__t))>;
    requires sizeof(ptrdiff_t) > sizeof(make_signed_t<decltype(ranges::size(__t))>);
    { static_cast<ptrdiff_t>(ranges::size(__t)) } -> integral;
  };

  template<class _Tp>
  concept __can_ssize_signed = requires (_Tp&& __t) {
    typename make_signed_t<decltype(ranges::size(__t))>;
    requires sizeof(ptrdiff_t) <= sizeof(make_signed_t<decltype(ranges::size(__t))>);
    { static_cast<make_signed_t<decltype(ranges::size(__t))>>(ranges::size(__t)) } -> integral;
  };
#else // ^^^ CXX20 ^^^ / vvv CXX17 vvv
  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __can_ssize_ptrdiff_,
    requires(_Tp&& __t)(
      typename(make_signed_t<decltype(ranges::size(__t))>),
      requires((sizeof(ptrdiff_t) > sizeof(make_signed_t<decltype(ranges::size(__t))>))),
      (static_cast<ptrdiff_t>(ranges::size(__t)))
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __can_ssize_ptrdiff = _LIBCUDACXX_FRAGMENT(__can_ssize_ptrdiff_, _Tp);

  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __can_ssize_signed_,
    requires(_Tp&& __t)(
      typename(make_signed_t<decltype(ranges::size(__t))>),
      requires(sizeof(ptrdiff_t) <= sizeof(make_signed_t<decltype(ranges::size(__t))>)),
      requires(integral<make_signed_t<decltype(ranges::size(__t))>>),
      (static_cast<make_signed_t<decltype(ranges::size(__t))>>(ranges::size(__t)))
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __can_ssize_signed = _LIBCUDACXX_FRAGMENT(__can_ssize_signed_, _Tp);
#endif // _LIBCUDACXX_STD_VER < 20

  struct __fn {
    _LIBCUDACXX_TEMPLATE(class _Tp)
      (requires __can_ssize_ptrdiff<_Tp>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY 
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(ranges::size(__t))) {
      return static_cast<ptrdiff_t>(ranges::size(__t));
    }
    
    _LIBCUDACXX_TEMPLATE(class _Tp)
      (requires __can_ssize_signed<_Tp>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY 
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(ranges::size(__t))) {
      using _Signed = make_signed_t<decltype(ranges::size(__t))>;
      return static_cast<_Signed>(ranges::size(__t));
    }
  };
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY _LIBCUDACXX_INLINE_VAR constexpr auto ssize = __ssize::__fn{};
} // namespace __cpo

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX___RANGES_SIZE_H
