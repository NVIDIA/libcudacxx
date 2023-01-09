// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___ITERATOR_PROJECTED_H
#define _LIBCUDACXX___ITERATOR_PROJECTED_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__iterator/concepts.h"
#include "../__iterator/incrementable_traits.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/remove_cvref.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 17

template<indirectly_readable _It, indirectly_regular_unary_invocable<_It> _Proj>
struct projected {
  using value_type = remove_cvref_t<indirect_result_t<_Proj&, _It>>;
  indirect_result_t<_Proj&, _It> operator*() const; // not defined
};

template<weakly_incrementable _It, class _Proj>
struct incrementable_traits<projected<_It, _Proj>> {
  using difference_type = iter_difference_t<_It>;
};

#elif _LIBCUDACXX_STD_VER > 14

template<class _It, class _Proj>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __projectable_,
  requires()(
    requires(indirectly_readable<_It>),
    requires(indirectly_regular_unary_invocable<_Proj, _It>)
  ));

template<class _It, class _Proj>
_LIBCUDACXX_CONCEPT __projectable = _LIBCUDACXX_FRAGMENT(__projectable_, _It, _Proj);

_LIBCUDACXX_TEMPLATE(class _It, class _Proj)
  (requires __projectable<_It, _Proj>)
struct projected {
  using value_type = remove_cvref_t<indirect_result_t<_Proj&, _It>>;
  _LIBCUDACXX_INLINE_VISIBILITY indirect_result_t<_Proj&, _It> operator*() const; // not defined
};

template<class _It, class _Proj>
struct incrementable_traits<projected<_It, _Proj>, enable_if_t<weakly_incrementable<_It>>> {
  using difference_type = iter_difference_t<_It>;
};

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_PROJECTED_H
