//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60

// <atomic>

// template <class T>
// struct atomic
// {
//     bool is_lock_free() const volatile;
//     bool is_lock_free() const;
//     void store(T desr, memory_order m = memory_order_seq_cst) volatile;
//     void store(T desr, memory_order m = memory_order_seq_cst);
//     T load(memory_order m = memory_order_seq_cst) const volatile;
//     T load(memory_order m = memory_order_seq_cst) const;
//     operator T() const volatile;
//     operator T() const;
//     T exchange(T desr, memory_order m = memory_order_seq_cst) volatile;
//     T exchange(T desr, memory_order m = memory_order_seq_cst);
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order s, memory_order f) volatile;
//     bool compare_exchange_weak(T& expc, T desr, memory_order s, memory_order f);
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order s, memory_order f) volatile;
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order s, memory_order f);
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order m = memory_order_seq_cst);
//     bool compare_exchange_strong(T& expc, T desr,
//                                 memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order m = memory_order_seq_cst);
//
//     atomic() = default;
//     constexpr atomic(T desr);
//     atomic(const atomic&) = delete;
//     atomic& operator=(const atomic&) = delete;
//     atomic& operator=(const atomic&) volatile = delete;
//     T operator=(T) volatile;
//     T operator=(T);
// };
//
// typedef atomic<bool> atomic_bool;

#include <atomic>
#include <cassert>

#include <cmpxchg_loop.h>

#include "test_macros.h"
#if !defined(TEST_COMPILER_C1XX)
  #include "placement_new.h"
#endif

template<template<cuda::thread_scope> typename Atomic, cuda::thread_scope Scope>
__host__ __device__
void do_test()
{
    {
        volatile Atomic<Scope> obj(true);
        assert(obj == true);
        bool b0 = obj.is_lock_free();
        (void)b0; // to placate scan-build
        obj.store(false);
        assert(obj == false);
        obj.store(true, std::memory_order_release);
        assert(obj == true);
        assert(obj.load() == true);
        assert(obj.load(std::memory_order_acquire) == true);
        assert(obj.exchange(false) == true);
        assert(obj == false);
        assert(obj.exchange(true, std::memory_order_relaxed) == false);
        assert(obj == true);
        bool x = obj;
        assert(cmpxchg_weak_loop(obj, x, false) == true);
        assert(obj == false);
        assert(x == true);
        assert(obj.compare_exchange_weak(x, true,
                                         std::memory_order_seq_cst) == false);
        assert(obj == false);
        assert(x == false);
        obj.store(true);
        x = true;
        assert(cmpxchg_weak_loop(obj, x, false,
                                 std::memory_order_seq_cst,
                                 std::memory_order_seq_cst) == true);
        assert(obj == false);
        assert(x == true);
        x = true;
        obj.store(true);
        assert(obj.compare_exchange_strong(x, false) == true);
        assert(obj == false);
        assert(x == true);
        assert(obj.compare_exchange_strong(x, true,
                                         std::memory_order_seq_cst) == false);
        assert(obj == false);
        assert(x == false);
        x = true;
        obj.store(true);
        assert(obj.compare_exchange_strong(x, false,
                                           std::memory_order_seq_cst,
                                           std::memory_order_seq_cst) == true);
        assert(obj == false);
        assert(x == true);
        assert((obj = false) == false);
        assert(obj == false);
        assert((obj = true) == true);
        assert(obj == true);
    }
    {
        Atomic<Scope> obj(true);
        assert(obj == true);
        bool b0 = obj.is_lock_free();
        (void)b0; // to placate scan-build
        obj.store(false);
        assert(obj == false);
        obj.store(true, std::memory_order_release);
        assert(obj == true);
        assert(obj.load() == true);
        assert(obj.load(std::memory_order_acquire) == true);
        assert(obj.exchange(false) == true);
        assert(obj == false);
        assert(obj.exchange(true, std::memory_order_relaxed) == false);
        assert(obj == true);
        bool x = obj;
        assert(cmpxchg_weak_loop(obj, x, false) == true);
        assert(obj == false);
        assert(x == true);
        assert(obj.compare_exchange_weak(x, true,
                                         std::memory_order_seq_cst) == false);
        assert(obj == false);
        assert(x == false);
        obj.store(true);
        x = true;
        assert(cmpxchg_weak_loop(obj, x, false,
                                 std::memory_order_seq_cst,
                                 std::memory_order_seq_cst) == true);
        assert(obj == false);
        assert(x == true);
        x = true;
        obj.store(true);
        assert(obj.compare_exchange_strong(x, false) == true);
        assert(obj == false);
        assert(x == true);
        assert(obj.compare_exchange_strong(x, true,
                                         std::memory_order_seq_cst) == false);
        assert(obj == false);
        assert(x == false);
        x = true;
        obj.store(true);
        assert(obj.compare_exchange_strong(x, false,
                                           std::memory_order_seq_cst,
                                           std::memory_order_seq_cst) == true);
        assert(obj == false);
        assert(x == true);
        assert((obj = false) == false);
        assert(obj == false);
        assert((obj = true) == true);
        assert(obj == true);
    }
    {
        Atomic<Scope> obj(true);
        assert(obj == true);
        bool b0 = obj.is_lock_free();
        (void)b0; // to placate scan-build
        obj.store(false);
        assert(obj == false);
        obj.store(true, std::memory_order_release);
        assert(obj == true);
        assert(obj.load() == true);
        assert(obj.load(std::memory_order_acquire) == true);
        assert(obj.exchange(false) == true);
        assert(obj == false);
        assert(obj.exchange(true, std::memory_order_relaxed) == false);
        assert(obj == true);
        bool x = obj;
        assert(cmpxchg_weak_loop(obj, x, false) == true);
        assert(obj == false);
        assert(x == true);
        assert(obj.compare_exchange_weak(x, true,
                                         std::memory_order_seq_cst) == false);
        assert(obj == false);
        assert(x == false);
        obj.store(true);
        x = true;
        assert(cmpxchg_weak_loop(obj, x, false,
                                 std::memory_order_seq_cst,
                                 std::memory_order_seq_cst) == true);
        assert(obj == false);
        assert(x == true);
        x = true;
        obj.store(true);
        assert(obj.compare_exchange_strong(x, false) == true);
        assert(obj == false);
        assert(x == true);
        assert(obj.compare_exchange_strong(x, true,
                                         std::memory_order_seq_cst) == false);
        assert(obj == false);
        assert(x == false);
        x = true;
        obj.store(true);
        assert(obj.compare_exchange_strong(x, false,
                                           std::memory_order_seq_cst,
                                           std::memory_order_seq_cst) == true);
        assert(obj == false);
        assert(x == true);
        assert((obj = false) == false);
        assert(obj == false);
        assert((obj = true) == true);
        assert(obj == true);
    }
    {
        typedef Atomic<Scope> A;
        TEST_ALIGNAS_TYPE(A) char storage[sizeof(A)] = {1};
        A& zero = *new (storage) A();
        assert(zero == false);
        zero.~A();
    }
}

template<cuda::thread_scope Scope>
using cuda_std_atomic = std::atomic<bool>;

template<cuda::thread_scope Scope>
using cuda_atomic = cuda::atomic<bool, Scope>;


int main(int, char**)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
    do_test<cuda_std_atomic, cuda::thread_scope_system>();
    do_test<cuda_atomic, cuda::thread_scope_system>();
#endif
    do_test<cuda_atomic, cuda::thread_scope_device>();
    do_test<cuda_atomic, cuda::thread_scope_block>();

  return 0;
}
