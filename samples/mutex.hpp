/*

Copyright (c) 2018, NVIDIA Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#ifndef _SIMT_MUTEX
#define _SIMT_MUTEX

#include <thread>

#include <simt/atomic>

#ifndef __builtin_expect
#define __builtin_expect(c,v) (c)
#endif

namespace simt { namespace experimental { inline namespace v1 {

using namespace simt::std;

/*
    gmutex - a general-purpose mutex for GPUs and their IO coprocessors

    Concepts:
        * lucked, an unfairly-held locked state
        * tocket, the ticket now being served

    This algorithm is a hybrid between a spinlock and a ticket lock. The lock 
    state is either unlocked, locked, or lucked. The lucked state is only available
    to be entered into when there are no outstanding tickets on the lock, which
    serve to communicate starvation between contending threads. Long-held tickets 
    cause some hysteresis, by pushing other threads to get tickets as well.
*/
template<class Int = unsigned>
struct basic_gmutex
{
    atomic<Int> __lock = ATOMIC_VAR_INIT(0); // combined state and tickets
    atomic<Int> __tock = ATOMIC_VAR_INIT(0); // "now serving" this ticket

    // components of the __lock state: states proper, and the ticket increment
    static constexpr Int __unlock_state = 0;
    static constexpr Int __locked_state = 1;
    static constexpr Int __lucked_state = __locked_state << 1;
    static constexpr Int __ticket_value = __lucked_state << 1;

    // for convenience
    static constexpr Int __state_mask   = __ticket_value - 1;
    static constexpr Int __ticket_mask  = ~__state_mask;

    __host__ __device__ inline void unlock() noexcept
    {
        // clear the lock state
        Int const old = __lock.fetch_and(__ticket_mask, memory_order_release);
        if((old & __lucked_state) == 0)
            // only when held fairly do we move the served ticket forward
            __tock.fetch_add(__ticket_value, memory_order_release);
    }
    __host__ __device__ inline void lock() noexcept
    {
        // try to lock unfairly, else switch to the ticket side to avoid starvation
        if (__builtin_expect(try_lock(), 1))
            return;
        __lock_slow();
    }
    __host__ __device__ inline bool try_lock() noexcept
    {
        auto tock = __tock.load(memory_order_relaxed);
        for (int i = 0; i < 64; ++i) {
            Int old = tock;
            // try to get it unfairly
            if(__builtin_expect(__lock.compare_exchange_weak(old, old | __lucked_state, memory_order_acquire, memory_order_relaxed),1))
                return true;
            // the lock can be held unfairly whenever it is unlocked, and ticket == tocket
            for (; old != tock && i < 64; ++i)
                __sleep(old, tock, old & __ticket_mask);
        }
        return false;
    }

    /*__host__ __device__*/ constexpr basic_gmutex() noexcept = default;
    basic_gmutex(const basic_gmutex&) = delete;
    basic_gmutex &operator=(const basic_gmutex&) = delete;

    __host__ __device__ void __lock_slow() noexcept
    {
        // get a ticket by increment -- this results in disabling the lucked state
        Int old = __lock.fetch_add(__ticket_value, memory_order_relaxed);
        Int const tick = old & __ticket_mask;
        Int tock = __tock.load(memory_order_relaxed);
        while(1) {
            // wait for your ticket to be served
            if(__builtin_expect(tock != tick,0)) {
                __sleep(old, tock, tick);
                continue;
            }
            old &= __ticket_mask;
            // obtain the lock fairly -- note this still needs a CAS because:
            // 1) the first transition from an unfair to a fair epoch is racy
            // 2) the ticket values keep changing and we want to merge into it
            if (__builtin_expect(__lock.compare_exchange_weak(old, old | __locked_state, memory_order_acquire, memory_order_relaxed),1))
                return;
        }
    }
    __host__ __device__ void __sleep(Int& old, Int& tock, Int tick) noexcept
    {
        // wait proportionally, then reload key values
#ifndef __CUDA_ARCH__
        ::std::this_thread::sleep_for(::std::chrono::nanoseconds((tick - tock) * 128));
#else
        __simt_nanosleep((tick - tock) * 128);
#endif
        tock = __tock.load(memory_order_acquire); // <-- omg, tricky
        old = __lock.load(memory_order_relaxed);
    }
};

using mutex = basic_gmutex<>;

}}}

#endif

