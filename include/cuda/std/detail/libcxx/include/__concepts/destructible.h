//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_DESTRUCTIBLE_H
#define _LIBCUDACXX___CONCEPTS_DESTRUCTIBLE_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__concepts/__concept_macros.h"
#include "../__type_traits/is_object.h"
#include "../__type_traits/is_nothrow_destructible.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 11

// [concept.destructible]

#if defined(_LIBCUDACXX_COMPILER_GCC) && _GNUC_VER < 1100

template<class _Tp>
_LIBCUDACXX_CONCEPT destructible = _LIBCUDACXX_TRAIT(is_nothrow_destructible, _Tp);

#else // ^^^ _LIBCUDACXX_COMPILER_GCC && GCC < 11 ^^^ / vvv other compilers vvv

#if _LIBCUDACXX_STD_VER > 17 // use concepts

template<class _Tp>
inline constexpr bool __destructible_impl_v = false;

template<class _Tp>
    requires requires(_Tp& __t) { { __t.~_Tp() } noexcept; }
inline constexpr bool __destructible_impl_v<_Tp> = true;

#else // ^^^ _LIBCUDACXX_STD_VER > 17 ^^^ / vvv _LIBCUDACXX_STD_VER < 20 vvv

template<class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __destructible_impl_req_,
  requires(_Tp& __t)(
    requires(_LIBCUDACXX_TRAIT(is_object, _Tp)),
    requires(noexcept(__t.~_Tp()))
  ));

template<class _Tp>
struct __destructible_impl : public _BoolConstant<_LIBCUDACXX_FRAGMENT(__destructible_impl_req_, _Tp)> {};

#if _LIBCUDACXX_STD_VER > 14
template<class _Tp>
inline constexpr bool __destructible_impl_v = _LIBCUDACXX_FRAGMENT(__destructible_impl_req_, _Tp);
#endif // _LIBCUDACXX_STD_VER > 14

#endif // _LIBCUDACXX_STD_VER < 20

#if _LIBCUDACXX_STD_VER > 14 // inline variables

template<class _Tp>
inline constexpr bool __destructible_v = __destructible_impl_v<_Tp>;

template<class _Tp>
inline constexpr bool __destructible_v<_Tp&> = true;

template<class _Tp>
inline constexpr bool __destructible_v<_Tp&&> = true;

template<class _Tp, size_t _Nm>
inline constexpr bool __destructible_v<_Tp[_Nm]> = __destructible_v<_Tp>;

#else // ^^^ _LIBCUDACXX_STD_VER > 14 ^^^ / vvv _LIBCUDACXX_STD_VER < 17 vvv

template<class _Tp>
struct __destructible : public __destructible_impl<_Tp> {};

template<class _Tp>
struct __destructible<_Tp&> : public true_type {};

template<class _Tp>
struct __destructible<_Tp&&> : public true_type {};

template<class _Tp, size_t _Nm>
struct __destructible<_Tp[_Nm]> : public __destructible<_Tp> {};

#endif // _LIBCUDACXX_STD_VER < 17

template<class _Tp>
_LIBCUDACXX_CONCEPT destructible = _LIBCUDACXX_TRAIT(__destructible, _Tp);

#endif // !_LIBCUDACXX_COMPILER_GCC || __GNUC__ > 11

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_DESTRUCTIBLE_H
