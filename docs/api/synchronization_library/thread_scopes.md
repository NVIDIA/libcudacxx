# Thread Scopes

Standard C++ presents a view that the cost to synchronize threads is uniform and low. CUDA C++ is different: the overhead is low within block, and high across arbitrary threads. Libcu++ introduces concept of thread scopes to the Standard's concurrency facilities.

```c++
namespace cuda {

enum thread_scope {
    thread_scope_system,
    thread_scope_device,
    thread_scope_block,
    thread_scope_thread
};

template<class T, thread_scope Scope>
class atomic;

void atomic_thread_fence(memory_order, thread_scope);

template<thread_scope Scope, class Completion>
class barrier;

template<thread_scope Scope>
class latch;

template<thread_scope Scope>
class binary_semaphore;

template<thread_scope Scope, ptrdiff_t LeastMaximum>
class counting_semaphore;

}
```

A program thread is related to another program thread by one or more thread scope:
1. Each thread is related to each other thread by the system thread scope, represented with `thread_scope_system`, for both CPU and GPU threads.
2. Each GPU thread is related to each other GPU thread in the same CUDA device by the device thread scope, represented with `thread_scope_device`.
3. Each GPU thread is related to each other GPU thread in the same CUDA block by the block thread scope, represented with `thread_scope_block`.

Refer to the CUDA programming guide for more information on how CUDA launches threads into devices and blocks.

## Extended consistency model

The Standard C++ memory model governs the semantics of libcu++, as if the following modifications were applied to its specification.

Add this paragraph to `[intro.races]` before paragraph 21:

> ***An atomic operation is atomic at the scope it specifies if:***
> 1. ***it specifies a scope other than `thread_scope_system`, or***
> 2. ***it effects an object in managed memory when `concurrentManagedAccess` is `1`, or***
> 3. ***it effects an object in host memory when `hostNativeAtomicSupported` is `1`, or***
> 4. ***it effects an object in peer memory and only GPU threads access it.***
>
> ***See the CUDA programming guide for more information on `concurrentManagedAccess` and `hostNativeAtomicSupported`.***

Extend `[intro.races]` paragraph 21 of N4860 as follows:
> The execution of a program contains a data race if it contains two potentially concurrent conflicting actions, at least one of which is not atomic ***at a scope that includes the thread that performed the other operation***, and neither happens before the other, except for the special case for signal handlers described below. Any such data race results in undefined behavior. [...]

Extend `[thread.barrier.class]` paragraph 4 of N4860 as follows:
> 4. Concurrent invocations of the member functions of barrier, other than its destructor, do not introduce data races ***as if they were atomic operations.***. [...]

Extend `[thread.latch.class]` paragraph 2 of N4860 as follows:
> 2. Concurrent invocations of the member functions of latch, other than its destructor, do not introduce data races ***as if they were atomic operations.***.

Extend `[thread.sema.cnt]` paragraph 3 of N4860 as follows:
> 3. Concurrent invocations of the member functions of counting_semaphore, other than its destructor, do not introduce data races ***as if they were atomic operations.***.

Extend `[atomics.fences]` paragraphs 2 through 4 of N4860 as follows:
> A release fence A synchronizes with an acquire fence B if there exist atomic operations X and Y, both operating on some atomic object M, such that A is sequenced before X, X modifies M, Y is sequenced before B, and Y reads the value written by X or a value written by any side effect in the hypothetical release sequence X would head if it were a release operation ***,and each operation (A, B, X, and Y) specifies a scope that includes the thread that performed each other operation***.

> A release fence A synchronizes with an atomic operation B that performs an acquire operation on an atomic object M if there exists an atomic operation X such that A is sequenced before X, X modifies M, and B reads the value written by X or a value written by any side effect in the hypothetical release sequence X would head if it were a release operation ***,and each operation (A, B, and X) specifies a scope that includes the thread that performed each other operation***.

> An atomic operation A that is a release operation on an atomic object M synchronizes with an acquire fence B if there exists some atomic operation X on M such that X is sequenced before B and reads the value written by A or a value written by any side effect in the release sequence headed by A ***,and each operation (A, B, and X) specifies a scope that includes the thread that performed each other operation***.
