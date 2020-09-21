# Thread Scopes

Concurrency facilities in Standard C++ (at least up to C++20) present a view that all threads have a uniform and low cost of synchronization to each other.

However, threads in CUDA C++ see a non-uniform cost of synchronization: it is low among threads in the same block, and high for arbitrary threads in the same system.

The most essential extension of libcu++ is the concept of thread scopes, a hierarchy of thread groupings, and specializations of class templates for thread scopes.

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

## Requirements

Extend `[intro.races]` paragraph 21 of N4860 as follows:

> The execution of a program contains a data race if it contains two potentially concurrent conflicting actions, at least one of which is not atomic ***with a scope that includes the thread that performed the other***, and neither happens before the other, except for the special case for signal handlers described below. Any such data race results in undefined behavior. [...]

Extend `[thread.barrier.class]` paragraph 4 of N4860 as follows:

> Concurrent invocations of the member functions of barrier, other than its destructor, do not introduce data races ***if performed by threads related by the scope specified by template argument***. [...]

Extend `[thread.latch.class]` paragraph 2 of N4860 as follows:

> Concurrent invocations of the member functions of latch, other than its destructor, do not introduce data races ***if performed by threads related by the scope specified by template argument***.

Extend `[thread.sema.cnt]` paragraph 3 of N4860 as follows:

> Concurrent invocations of the member functions of counting_semaphore, other than its destructor, do not introduce data races ***if performed by threads related by the scope specified by template argument***.

Extend `[atomics.fences]` paragraph 2 of N4860 as follows:

> A release fence A synchronizes with an acquire fence B if there exist atomic operations X and Y, both operating on some atomic object M, such that A is sequenced before X, X modifies M, Y is sequenced before B, and Y reads the value written by X or a value written by any side effect in the hypothetical release sequence X would head if it were a release operation ***,and each operation (A, B, X, and Y) has a scope that includes the thread that performed each other operation***.

And paragraph 3:

> A release fence A synchronizes with an atomic operation B that performs an acquire operation on an atomic object M if there exists an atomic operation X such that A is sequenced before X, X modifies M, and B reads the value written by X or a value written by any side effect in the hypothetical release sequence X would head if it were a release operation ***,and each operation (A, B, and X) has a scope that includes the thread that performed each other operation***.

And paragraph 4:

> An atomic operation A that is a release operation on an atomic object M synchronizes with an acquire fence B if there exists some atomic operation X on M such that X is sequenced before B and reads the value written by A or a value written by any side effect in the release sequence headed by A ***,and each operation (A, B, and X) has a scope that includes the thread that performed each other operation***.

## Restrictions

In addition to meeting the requirements above, programs shall ensure that an object of type `atomic`, `barrier`, `latch`, `binary_semaphore` or `counting_semaphore` is not accessed concurrently by CPU and GPU threads unless:
  1. it is in managed memory, with `concurrentManagedAccess==1`, or
  2. it is in host memory, with `hostNativeAtomicSupported==1`.
