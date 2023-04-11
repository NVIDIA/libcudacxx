//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef __LIBCUDACXX_ATOMIC_CUDA_LOCAL_H
#define __LIBCUDACXX_ATOMIC_CUDA_LOCAL_H

__device__ static inline int __memcmp(const void* __lhs, const void* __rhs, size_t __count)
{
  const unsigned char * __l = (const unsigned char*)__lhs;
  const unsigned char * __r = (const unsigned char*)__rhs;
  for (size_t __i = 0; __i < __count; ++__i)
    if (__l[__i] != __r[__i]) return __l[__i] < __r[__i]? -1 : 1;
  return 0;
}

template <class _Type>
__device__ bool __cuda_load_weak_if_local(const volatile _Type* __ptr, _Type* __ret)
{ 
  if (!__isLocal((const void*)__ptr)) return false;
  memcpy((void*)__ret, (void const *)__ptr, sizeof(_Type));
  __nanosleep(0);
  return true;
}

template <class _Type>
__device__ bool __cuda_store_weak_if_local(volatile _Type* __ptr, _Type* __val)
{
  if (!__isLocal((const void*)__ptr)) return false;
  memcpy((void*)__ptr, (void const*)__val, sizeof(_Type));
  return true;
}

template <class _Type>
__device__ bool __cuda_compare_exchange_weak_if_local(volatile _Type *__ptr, _Type *__expected,
							const _Type *__desired, bool* __success)
{ 
  if (!__isLocal((const void*)__ptr)) return false;
  if (0 == __memcmp((const void*)__ptr, (const void*)__expected, sizeof(_Type))) {
    memcpy((void*)__ptr,  (void const*)__desired, sizeof(_Type));
    *__success = true;
  } else {
    memcpy((void*)__expected,  (void const*)__ptr, sizeof(_Type));
    *__success = false;
  }
  __nanosleep(0);
  return true;
}

template <class _Type>
__device__ bool __cuda_exchange_weak_if_local(volatile _Type *__ptr, _Type *__val, _Type *__ret)
{ 
  if (!__isLocal((const void*)__ptr)) return false;
  memcpy((void*)__ret, (void const*)__ptr, sizeof(_Type));
  memcpy((void*)__ptr, (void const*)__val, sizeof(_Type));
  __nanosleep(0);
  return true;
}

template <class _Type, class _BOp>
__device__ bool __cuda_fetch_weak_if_local(volatile _Type *__ptr, _Type __val, _Type* __ret, _BOp&& __bop)
{ 
  if (!__isLocal((const void*)__ptr)) return false;
  memcpy((void*)__ret, (void const*)__ptr, sizeof(_Type));
  __bop(*__ptr, __val);
  __nanosleep(0);
  return true;
}

template <class _Type>
__device__ bool __cuda_fetch_and_weak_if_local(volatile _Type *__ptr, _Type __val, _Type* __ret)
{
  return __cuda_fetch_weak_if_local(__ptr, __val, __ret, [](volatile _Type& out, _Type const& in) { out &= in; });
}

template <class _Type>
__device__ bool __cuda_fetch_or_weak_if_local(volatile _Type *__ptr, _Type __val, _Type* __ret)
{
  return __cuda_fetch_weak_if_local(__ptr, __val, __ret, [](volatile _Type& out, _Type const& in) { out |= in; });
}

template <class _Type>
__device__ bool __cuda_fetch_xor_weak_if_local(volatile _Type *__ptr, _Type __val, _Type* __ret)
{
  return __cuda_fetch_weak_if_local(__ptr, __val, __ret, [](volatile _Type& out, _Type const& in) { out ^= in; });
}

template <class _Type>
__device__ bool __cuda_fetch_add_weak_if_local(volatile _Type *__ptr, _Type __val, _Type* __ret)
{
  return __cuda_fetch_weak_if_local(__ptr, __val, __ret, [](volatile _Type& out, _Type const& in) { out += in; });
}

template <class _Type>
__device__ bool __cuda_fetch_sub_weak_if_local(volatile _Type *__ptr, _Type __val, _Type* __ret)
{
  return __cuda_fetch_weak_if_local(__ptr, __val, __ret, [](volatile _Type& out, _Type const& in) { out -= in; });
}

template <class _Type>
__device__ bool __cuda_fetch_max_weak_if_local(volatile _Type *__ptr, _Type __val, _Type* __ret)
{
  return __cuda_fetch_weak_if_local(__ptr, __val, __ret, [](volatile _Type& out, _Type const& in) { out = out < in? in : out; });
}

template <class _Type>
__device__ bool __cuda_fetch_min_weak_if_local(volatile _Type *__ptr, _Type __val, _Type* __ret)
{
  return __cuda_fetch_weak_if_local(__ptr, __val, __ret, [](volatile _Type& out, _Type const& in) { out = in < out? in : out; });
}

#endif
