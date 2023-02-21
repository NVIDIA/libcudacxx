// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANGES_COPYABLE_BOX_H
#define _LIBCUDACXX___RANGES_COPYABLE_BOX_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__concepts/constructible.h"
#include "../__concepts/copyable.h"
#include "../__concepts/movable.h"
#include "../__memory/addressof.h"
#include "../__memory/construct_at.h"
#include "../__ranges/access.h"
#include "../__ranges/concepts.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_constructible.h"
#include "../__type_traits/is_nothrow_constructible.h"
#include "../__type_traits/is_nothrow_copy_constructible.h"
#include "../__type_traits/is_nothrow_default_constructible.h"
#include "../__type_traits/is_nothrow_move_constructible.h"
#include "../__type_traits/is_object.h"
#include "../__utility/forward.h"
#include "../__utility/in_place.h"
#include "../__utility/move.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _LIBCUDACXX_STD_VER > 14

// __copyable_box allows turning a type that is copy-constructible (but maybe not copy-assignable) into
// a type that is both copy-constructible and copy-assignable. It does that by introducing an empty state
// and basically doing destroy-then-copy-construct in the assignment operator. The empty state is necessary
// to handle the case where the copy construction fails after destroying the object.
//
// In some cases, we can completely avoid the use of an empty state; we provide a specialization of
// __copyable_box that does this, see below for the details.
#if _LIBCUDACXX_STD_VER > 17
  template<class _Tp>
  concept __copy_constructible_object = copy_constructible<_Tp> && is_object_v<_Tp>;
#else
  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __copy_constructible_object_,
    requires()(
      requires(copy_constructible<_Tp>),
      requires(is_object_v<_Tp>)
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __copy_constructible_object = _LIBCUDACXX_FRAGMENT(__copy_constructible_object_, _Tp);
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

  template<class _Tp, class = void>
  struct __copyable_box_destruct_base {
    union
    {
        _LIBCUDACXX_NO_UNIQUE_ADDRESS _Tp __val_;
    };
    bool __engaged_ = false;

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __copyable_box_destruct_base() noexcept {};

    template<class ..._Args>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __copyable_box_destruct_base(in_place_t, _Args&& ...__args)
      noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
      : __val_(_CUDA_VSTD::forward<_Args>(__args)...), __engaged_(true)
    { }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 ~__copyable_box_destruct_base() noexcept {
        if (__engaged_) {
            __val_.~_Tp();
        }
    }

#if _LIBCUDACXX_STD_VER > 17
    // clang has issues with the ordering of defaulted SMF so the order here is important
    ~__copyable_box_destruct_base() requires is_trivially_destructible_v<_Tp> = default;
#endif
  };

  template<class _Tp>
  struct __copyable_box_destruct_base<_Tp, enable_if_t<is_trivially_destructible_v<_Tp>>> {
    union
    {
        _LIBCUDACXX_NO_UNIQUE_ADDRESS _Tp __val_;
    };
    bool __engaged_ = false;

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __copyable_box_destruct_base() noexcept {};

    template<class ..._Args>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __copyable_box_destruct_base(in_place_t, _Args&& ...__args)
      noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
      : __val_(_CUDA_VSTD::forward<_Args>(__args)...), __engaged_(true)
    { }
  };

  // Primary template - uses _CUDA_VSTD::optional and introduces an empty state in case assignment fails.
#if _LIBCUDACXX_STD_VER > 17
  template<__copy_constructible_object _Tp>
  class __copyable_box : public __copyable_box_destruct_base<_Tp> {
#else
  template<class _Tp, class = void, class = void, class = void>
  class __copyable_box;

  template<class _Tp>
  class __copyable_box<_Tp, enable_if_t<__copy_constructible_object<_Tp>>> : public __copyable_box_destruct_base<_Tp> {
#endif
  using __base = __copyable_box_destruct_base<_Tp>;

  public:
    _LIBCUDACXX_TEMPLATE(class ..._Args)
      (requires is_constructible_v<_Tp, _Args...>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr explicit __copyable_box(in_place_t, _Args&& ...__args)
      noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
      : __base(in_place, _CUDA_VSTD::forward<_Args>(__args)...)
    { }

    _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
      (requires default_initializable<_Tp2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __copyable_box() noexcept(is_nothrow_default_constructible_v<_Tp>)
      : __base(in_place)
    { }

#if _LIBCUDACXX_STD_VER > 17
    __copyable_box(__copyable_box const&) requires is_trivially_copy_constructible_v<_Tp> = default;
    __copyable_box(__copyable_box&&) requires is_trivially_move_constructible_v<_Tp> = default;
#endif

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 __copyable_box(const __copyable_box& __other)
      noexcept(is_nothrow_copy_constructible_v<_Tp>) : __base() {
      if (__other.__engaged_) {
        _CUDA_VSTD::construct_at(_CUDA_VSTD::addressof(this->__val_), __other.__val_);
        this->__engaged_ = true;
      }
    }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 __copyable_box(__copyable_box&& __other)
      noexcept(is_nothrow_move_constructible_v<_Tp>) : __base()  {
      if (__other.__engaged_) {
        _CUDA_VSTD::construct_at(_CUDA_VSTD::addressof(this->__val_), _CUDA_VSTD::move(__other.__val_));
        this->__engaged_ = true;
      }
    }

#if _LIBCUDACXX_STD_VER > 17
    __copyable_box& operator=(const __copyable_box&)
      requires copyable<_Tp> && is_trivially_copy_assignable_v<_Tp> = default;
#endif

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 __copyable_box& operator=(__copyable_box const& __other)
      noexcept(is_nothrow_copy_constructible_v<_Tp> && is_nothrow_copy_assignable_v<_Tp>)
    {
      if (this != _CUDA_VSTD::addressof(__other)) {
        if constexpr (copyable<_Tp>) {
          if (this->__engaged_) {
            if (__other.__engaged_) {
              this->__val_ = __other.__val_;
            } else {
              this->__val_.~_Tp();
              this->__engaged_ = false;
            }
          } else {
            if (__other.__engaged_) {
              _CUDA_VSTD::construct_at(_CUDA_VSTD::addressof(this->__val_), __other.__val_);
              this->__engaged_ = true;
            } else {
              /* nothing to do */
            }
          }
        } else {
          if (this->__engaged_) {
            this->__val_.~_Tp();
            this->__engaged_ = false;
          }
          if (__other.__engaged_) {
            _CUDA_VSTD::construct_at(_CUDA_VSTD::addressof(this->__val_), __other.__val_);
            this->__engaged_ = true;
          }
        }
      }
      return *this;
    }

#if _LIBCUDACXX_STD_VER > 17
    __copyable_box& operator=(__copyable_box&&)
      requires movable<_Tp> && is_trivially_move_assignable_v<_Tp> = default;
#endif

    _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
      (requires movable<_Tp2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 __copyable_box& operator=(__copyable_box&& __other)
      noexcept(is_nothrow_move_constructible_v<_Tp2>&& is_nothrow_move_assignable_v<_Tp2>)
    {
      if (this != _CUDA_VSTD::addressof(__other)) {
        if (this->__engaged_) {
          if (__other.__engaged_) {
            this->__val_ = _CUDA_VSTD::move(__other.__val_);
          } else {
            this->__val_.~_Tp();
            this->__engaged_ = false;
          }
        } else {
          if (__other.__engaged_) {
            _CUDA_VSTD::construct_at(_CUDA_VSTD::addressof(this->__val_), _CUDA_VSTD::move(__other.__val_));
            this->__engaged_ = true;
          } else {
            /* nothing to do */
          }
        }
      }
      return *this;
    }

    _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
      (requires (!movable<_Tp2>))
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 __copyable_box& operator=(__copyable_box&& __other)
      noexcept(is_nothrow_move_constructible_v<_Tp>)
    {
      if (this != _CUDA_VSTD::addressof(__other)) {
        if (this->__engaged_) {
          this->__val_.~_Tp();
          this->__engaged_ = false;
        }
        if (__other.__engaged_) {
          _CUDA_VSTD::construct_at(_CUDA_VSTD::addressof(this->__val_), _CUDA_VSTD::move(__other.__val_));
          this->__engaged_ = true;
        }
      }
      return *this;
    }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp const& operator*() const noexcept { return this->__val_; }
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp& operator*() noexcept { return this->__val_; }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr const _Tp *operator->() const noexcept { return _CUDA_VSTD::addressof(this->__val_); }
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp *operator->() noexcept { return _CUDA_VSTD::addressof(this->__val_); }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr bool __has_value() const noexcept { return this->__engaged_; }
  };

  // This partial specialization implements an optimization for when we know we don't need to store
  // an empty state to represent failure to perform an assignment. For copy-assignment, this happens:
  //
  // 1. If the type is copyable (which includes copy-assignment), we can use the type's own assignment operator
  //    directly and avoid using _CUDA_VSTD::optional.
  // 2. If the type is not copyable, but it is nothrow-copy-constructible, then we can implement assignment as
  //    destroy-and-then-construct and we know it will never fail, so we don't need an empty state.
  //
  // The exact same reasoning can be applied for move-assignment, with copyable replaced by movable and
  // nothrow-copy-constructible replaced by nothrow-move-constructible. This specialization is enabled
  // whenever we can apply any of these optimizations for both the copy assignment and the move assignment
  // operator.
  template<class _Tp>
  _LIBCUDACXX_CONCEPT __doesnt_need_empty_state_for_copy = copyable<_Tp> || is_nothrow_copy_constructible_v<_Tp>;

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __doesnt_need_empty_state_for_move = movable<_Tp> || is_nothrow_move_constructible_v<_Tp>;

#if _LIBCUDACXX_STD_VER > 17
  template<__copy_constructible_object _Tp>
    requires __doesnt_need_empty_state_for_copy<_Tp> && __doesnt_need_empty_state_for_move<_Tp>
  class __copyable_box<_Tp> {
#else
  template<class _Tp>
  class __copyable_box<_Tp, enable_if_t<__copy_constructible_object<_Tp>, nullptr_t>,
                            enable_if_t<__doesnt_need_empty_state_for_copy<_Tp>>,
                            enable_if_t<__doesnt_need_empty_state_for_move<_Tp>>> {
#endif
    _LIBCUDACXX_NO_UNIQUE_ADDRESS _Tp __val_;

  public:
    _LIBCUDACXX_TEMPLATE(class ..._Args)
      (requires is_constructible_v<_Tp, _Args...>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr explicit __copyable_box(in_place_t, _Args&& ...__args)
      noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
      : __val_(_CUDA_VSTD::forward<_Args>(__args)...)
    { }

    _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
      (requires default_initializable<_Tp2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __copyable_box() noexcept(is_nothrow_default_constructible_v<_Tp>)
      : __val_()
    { }

    __copyable_box(__copyable_box const&) = default;
    __copyable_box(__copyable_box&&) = default;


    // Implementation of assignment operators in case we perform optimization (1)
#if _LIBCUDACXX_STD_VER > 17
    __copyable_box& operator=(__copyable_box const&) requires copyable<_Tp> = default;
#else
    _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
      (requires copyable<_Tp2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __copyable_box& operator=(__copyable_box const& __other)
      noexcept(is_nothrow_copy_assignable_v<_Tp>) {
      if (this != _CUDA_VSTD::addressof(__other)) {
        __val_ = __other._val;
      }
      return *this;
    }
#endif

    // Implementation of assignment operators in case we perform optimization (2)
    _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
      (requires copy_constructible<_Tp2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 __copyable_box& operator=(__copyable_box const& __other) noexcept {
      static_assert(is_nothrow_copy_constructible_v<_Tp>);
      if (this != _CUDA_VSTD::addressof(__other)) {
        __val_.~_Tp();
        _CUDA_VSTD::construct_at(_CUDA_VSTD::addressof(__val_), __other.__val_);
      }
      return *this;
    }


#if _LIBCUDACXX_STD_VER > 17
    __copyable_box& operator=(__copyable_box&&) requires movable<_Tp> = default;
#else
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 __copyable_box& operator=(__copyable_box&& __other)
      noexcept(is_nothrow_move_assignable_v<_Tp>) {
      if (this != _CUDA_VSTD::addressof(__other)) {
        __val_ = _CUDA_VSTD::move(__other._val);
      }
      return *this;
    }
#endif

    _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
      (requires (!movable<_Tp2>))
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __copyable_box& operator=(__copyable_box&& __other) noexcept {
      static_assert(is_nothrow_move_constructible_v<_Tp>);
      if (this != _CUDA_VSTD::addressof(__other)) {
        __val_.~_Tp();
        _CUDA_VSTD::construct_at(_CUDA_VSTD::addressof(__val_), _CUDA_VSTD::move(__other.__val_));
      }
      return *this;
    }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp const& operator*() const noexcept { return __val_; }
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp& operator*() noexcept { return __val_; }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr const _Tp *operator->() const noexcept { return _CUDA_VSTD::addressof(__val_); }
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp *operator->() noexcept { return _CUDA_VSTD::addressof(__val_); }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr bool __has_value() const noexcept { return true; }
  };
  _LIBCUDACXX_END_NAMESPACE_RANGES_ABI

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX___RANGES_COPYABLE_BOX_H
