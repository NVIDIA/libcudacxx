//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_RANGES_MIN_H
#define _LIBCUDACXX___ALGORITHM_RANGES_MIN_H

#ifndef __cuda_std__
#include <__config>
#include <initializer_list>
#endif // __cuda_std__

#include "../__algorithm/ranges_min_element.h"
//#include "../__assert"
#include "../__concepts/copyable.h"
#include "../__functional/identity.h"
#include "../__functional/invoke.h"
#include "../__functional/ranges_operations.h"
#include "../__iterator/concepts.h"
#include "../__iterator/projected.h"
#include "../__ranges/access.h"
#include "../__ranges/concepts.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#if _LIBCUDACXX_STD_VER > 14

#if defined(_LIBCUDACXX_PUSH_MACROS)
  _LIBCUDACXX_PUSH_MACROS
#endif
#include "../__undef_macros"

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__min)

struct __fn {
  _LIBCUDACXX_TEMPLATE(class _Tp, class _Proj = identity, class _Comp = _CUDA_VRANGES::less)
    (requires indirect_strict_weak_order<_Comp, projected<const _Tp*, _Proj>>)
  _LIBCUDACXX_NODISCARD_EXT _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  const _Tp& operator()(const _Tp& __a, const _Tp& __b, _Comp __comp = {}, _Proj __proj = {}) const {
    return _CUDA_VSTD::invoke(__comp, _CUDA_VSTD::invoke(__proj, __b), _CUDA_VSTD::invoke(__proj, __a)) ? __b : __a;
  }

  _LIBCUDACXX_TEMPLATE(class _Tp, class _Proj = identity, class _Comp = _CUDA_VRANGES::less)
    (requires indirect_strict_weak_order<_Comp, projected<const _Tp*, _Proj>>)
  _LIBCUDACXX_NODISCARD_EXT _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  _Tp operator()(initializer_list<_Tp> __il, _Comp __comp = {}, _Proj __proj = {}) const {
    _LIBCUDACXX_ASSERT(__il.begin() != __il.end(), "initializer_list must contain at least one element");
    return *_CUDA_VRANGES::__min_element_impl(__il.begin(), __il.end(), __comp, __proj);
  }

  _LIBCUDACXX_TEMPLATE(class _Rp, class _Proj = identity, class _Comp = _CUDA_VRANGES::less)
    (requires input_range<_Rp> _LIBCUDACXX_AND 
              indirect_strict_weak_order<_Comp, projected<iterator_t<_Rp>, _Proj>> _LIBCUDACXX_AND
              indirectly_copyable_storable<iterator_t<_Rp>, range_value_t<_Rp>*>)
  _LIBCUDACXX_NODISCARD_EXT _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  range_value_t<_Rp> operator()(_Rp&& __r, _Comp __comp = {}, _Proj __proj = {}) const {
    auto __first = _CUDA_VRANGES::begin(__r);
    auto __last = _CUDA_VRANGES::end(__r);

    _LIBCUDACXX_ASSERT(__first != __last, "range must contain at least one element");

    if constexpr (forward_range<_Rp>) {
      return *_CUDA_VRANGES::__min_element_impl(__first, __last, __comp, __proj);
    } else {
      range_value_t<_Rp> __result = *__first;
      while (++__first != __last) {
        if (_CUDA_VSTD::invoke(__comp, _CUDA_VSTD::invoke(__proj, *__first), _CUDA_VSTD::invoke(__proj, __result)))
          __result = *__first;
      }
      return __result;
    }
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY inline constexpr auto min = __min::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_RANGES

#if defined(_LIBCUDACXX_POP_MACROS)
  _LIBCUDACXX_POP_MACROS
#endif

#endif // _LIBCUDACXX_STD_VER > 14

#endif // _LIBCUDACXX___ALGORITHM_RANGES_MIN_H
