// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_VIEW_INTERFACE_H
#define _LIBCUDACXX___RANGES_VIEW_INTERFACE_H

#ifndef __cuda_std__
#include <__config>
#include <cassert>
#endif // __cuda_std__

#include "../__concepts/derived_from.h"
#include "../__concepts/same_as.h"
#include "../__iterator/concepts.h"
#include "../__iterator/iterator_traits.h"
#include "../__iterator/prev.h"
#include "../__memory/pointer_traits.h"
#include "../__ranges/access.h"
#include "../__ranges/concepts.h"
#include "../__ranges/empty.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_class.h"
#include "../__type_traits/is_reference.h"
#include "../__type_traits/remove_cv.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

#if _LIBCUDACXX_STD_VER > 17
template<class _Derived>
  requires is_class_v<_Derived> && same_as<_Derived, remove_cv_t<_Derived>>
#else
template<class _Derived, 
  enable_if_t<is_class_v<_Derived> && same_as<_Derived, remove_cv_t<_Derived>>, int>>
#endif
class view_interface {
  _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _Derived& __derived() noexcept {
    static_assert(sizeof(_Derived) && derived_from<_Derived, view_interface> && view<_Derived>, "");
    return static_cast<_Derived&>(*this);
  }

  _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _Derived const& __derived() const noexcept {
    static_assert(sizeof(_Derived) && derived_from<_Derived, view_interface> && view<_Derived>, "");
    return static_cast<_Derived const&>(*this);
  }

public:
  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
    (requires forward_range<_D2>)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY constexpr bool empty()
  {
    return ranges::begin(__derived()) == ranges::end(__derived());
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
    (requires forward_range<const _D2>)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY constexpr bool empty() const
  {
    return ranges::begin(__derived()) == ranges::end(__derived());
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
    (requires invocable<ranges::__empty::__fn, _D2&>)
  _LIBCUDACXX_INLINE_VISIBILITY
  constexpr explicit operator bool()
  {
    return !ranges::empty(__derived());
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
    (requires invocable<ranges::__empty::__fn, const _D2&>)
  _LIBCUDACXX_INLINE_VISIBILITY
  constexpr explicit operator bool() const
  {
    return !ranges::empty(__derived());
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
    (requires contiguous_iterator<iterator_t<_D2>>)
  _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto data()
  {
    return _CUDA_VSTD::to_address(ranges::begin(__derived()));
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
    (requires range<const _D2> _LIBCUDACXX_AND contiguous_iterator<iterator_t<const _D2>>)
  _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto data() const
  {
    return _CUDA_VSTD::to_address(ranges::begin(__derived()));
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
    (requires forward_range<_D2> _LIBCUDACXX_AND sized_sentinel_for<sentinel_t<_D2>, iterator_t<_D2>>)
  _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto size()
  {
    return ranges::end(__derived()) - ranges::begin(__derived());
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
    (requires forward_range<const _D2> _LIBCUDACXX_AND sized_sentinel_for<sentinel_t<const _D2>, iterator_t<const _D2>>)
  _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto size() const
  {
    return ranges::end(__derived()) - ranges::begin(__derived());
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
    (requires forward_range<_D2>)
  _LIBCUDACXX_INLINE_VISIBILITY
  constexpr decltype(auto) front()
  {
    _LIBCUDACXX_ASSERT(!empty(),
        "Precondition `!empty()` not satisfied. `.front()` called on an empty view.");
    return *ranges::begin(__derived());
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
    (requires forward_range<const _D2>)
  _LIBCUDACXX_INLINE_VISIBILITY
  constexpr decltype(auto) front() const
  {
    _LIBCUDACXX_ASSERT(!empty(),
        "Precondition `!empty()` not satisfied. `.front()` called on an empty view.");
    return *ranges::begin(__derived());
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
    (requires bidirectional_range<_D2> _LIBCUDACXX_AND common_range<_D2>)
  _LIBCUDACXX_INLINE_VISIBILITY
  constexpr decltype(auto) back()
  {
    _LIBCUDACXX_ASSERT(!empty(),
        "Precondition `!empty()` not satisfied. `.back()` called on an empty view.");
    return *ranges::prev(ranges::end(__derived()));
  }

  _LIBCUDACXX_TEMPLATE(class _D2 = _Derived)
    (requires bidirectional_range<const _D2> _LIBCUDACXX_AND common_range<const _D2>)
  _LIBCUDACXX_INLINE_VISIBILITY
  constexpr decltype(auto) back() const
  {
    _LIBCUDACXX_ASSERT(!empty(),
        "Precondition `!empty()` not satisfied. `.back()` called on an empty view.");
    return *ranges::prev(ranges::end(__derived()));
  }

  _LIBCUDACXX_TEMPLATE(class _RARange = _Derived)
    (requires random_access_range<_RARange>)
  _LIBCUDACXX_INLINE_VISIBILITY
  constexpr decltype(auto) operator[](range_difference_t<_RARange> __index)
  {
    return ranges::begin(__derived())[__index];
  }

  _LIBCUDACXX_TEMPLATE(class _RARange = const _Derived)
    (requires random_access_range<_RARange>)
  _LIBCUDACXX_INLINE_VISIBILITY
  constexpr decltype(auto) operator[](range_difference_t<_RARange> __index) const
  {
    return ranges::begin(__derived())[__index];
  }
};

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX___RANGES_VIEW_INTERFACE_H
