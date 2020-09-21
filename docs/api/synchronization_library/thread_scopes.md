# Thread Scopes

Standard C++ presents a view that the cost to synchronize threads is uniform and low. CUDA C++ is different: the overhead is low among threads within a block, and high across arbitrary threads in the system.

To bridge these two realities, libcu++ introduces the concept of thread scopes to the Standard's concurrency facilities in the `cuda::` namespace, while retaining the syntax and semantics of Standard C++ by default.

## Synopsis

```c++
namespace cuda {

enum thread_scope {
    thread_scope_system,
    thread_scope_device,
    thread_scope_block,
    thread_scope_thread
};

template<class T, thread_scope Scope = thread_scope_thread>
class atomic;

void atomic_thread_fence(memory_order, thread_scope = thread_scope_thread);

template<thread_scope Scope = thread_scope_thread, 
         class Completion = /*implementation-defined*/>
class barrier;

template<thread_scope Scope = thread_scope_thread>
class latch;

template<thread_scope Scope = thread_scope_thread>
class binary_semaphore;

template<thread_scope Scope = thread_scope_thread, 
         ptrdiff_t LeastMaximum = /*implementation-defined*/>
class counting_semaphore;

}
```

## Scope relationships

Each program thread is related to each other program thread by one or more thread scope relations:
1. Each thread (CPU or GPU) is related to each other thread in the computer system by the *system* thread scope, specified with `thread_scope_system`.
2. Each GPU thread is related to each other GPU thread in the same CUDA device by the *device* thread scope, specified with `thread_scope_device`.
3. Each GPU thread is related to each other GPU thread in the same CUDA block by the *block* thread scope, specified with `thread_scope_block`.
4. Each thread (CPU or GPU) is related to itself by the `thread` thread scope, specified with `thread_scope_thread`.

Refer to the CUDA programming guide for more information on how CUDA launches threads into devices and blocks.

## Atomicity

An atomic operation is atomic at the scope it specifies if:
1. it specifies a scope other than `thread_scope_system`, or
2. it effects an object in managed memory with `concurrentManagedAccess` is `1`, or***
3. it effects an object in host memory with `hostNativeAtomicSupported` is `1`, or
4. it effects an object in peer memory and only GPU threads access it.

Refer to the CUDA programming guide for more information on managed memory, the `concurrentManagedAccess` property, host memory, the `hostNativeAtomicSupported` property, and peer memory.

## Data races

Modify `[intro.races]` paragraph 21 of N4860 as follows:
> The execution of a program contains a data race if it contains two potentially concurrent conflicting actions, at least one of which is not atomic ***at a scope that includes the thread that performed the other operation***, and neither happens before the other, except for the special case for signal handlers described below. Any such data race results in undefined behavior. [...]

Modify `[thread.barrier.class]` paragraph 4 of N4860 as follows:
> 4. Concurrent invocations of the member functions of barrier, other than its destructor, do not introduce data races ***as if they were atomic operations.***. [...]

Modify `[thread.latch.class]` paragraph 2 of N4860 as follows:
> 2. Concurrent invocations of the member functions of latch, other than its destructor, do not introduce data races ***as if they were atomic operations.***.

Modify `[thread.sema.cnt]` paragraph 3 of N4860 as follows:
> 3. Concurrent invocations of the member functions of counting_semaphore, other than its destructor, do not introduce data races ***as if they were atomic operations.***.

Modify `[atomics.fences]` paragraphs 2 through 4 of N4860 as follows:
> A release fence A synchronizes with an acquire fence B if there exist atomic operations X and Y, both operating on some atomic object M, such that A is sequenced before X, X modifies M, Y is sequenced before B, and Y reads the value written by X or a value written by any side effect in the hypothetical release sequence X would head if it were a release operation ***,and each operation (A, B, X, and Y) specifies a scope that includes the thread that performed each other operation***.

> A release fence A synchronizes with an atomic operation B that performs an acquire operation on an atomic object M if there exists an atomic operation X such that A is sequenced before X, X modifies M, and B reads the value written by X or a value written by any side effect in the hypothetical release sequence X would head if it were a release operation ***,and each operation (A, B, and X) specifies a scope that includes the thread that performed each other operation***.

> An atomic operation A that is a release operation on an atomic object M synchronizes with an acquire fence B if there exists some atomic operation X on M such that X is sequenced before B and reads the value written by A or a value written by any side effect in the release sequence headed by A ***,and each operation (A, B, and X) specifies a scope that includes the thread that performed each other operation***.
