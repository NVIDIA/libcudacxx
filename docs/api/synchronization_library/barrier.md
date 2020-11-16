---
grand_parent: API
parent: Synchronization Library
nav_order: 2
---

# `<cuda/std/barrier>`, `<cuda/barrier>`

## Extensions

The class template `barrier` takes an additional [thread scope] argument,
  defaulted to `thread_scope_system`.

```c++
// This barrier is suitable for all threads in the system.
cuda::barrier<cuda::thread_scope_system> a;

// These barriers have the same type as the previous one.
cuda::barrier<> ba;
cuda::std::barrier<> bb;

// This barrier is suitable for all threads in the same thread block.
cuda::barrier<cuda::thread_scope_block> c;
```

The class template `barrier` may also be declared without initialization in the `cuda::` namespace; a
friend function `init` may be used to initialize the object.

```c++
// Shared memory does not allow initialization.
__shared__ cuda::barrier<cuda::thread_scope_block> b;

init(&b, 1); // Use this friend function to initialize the object.
/*
namespace cuda {
  template<thread_scope Sco, class CompletionF>
  __host__ __device__ void init(barrier<Sco,CompletionF>* bar, std::ptrdiff_t expected);
  template<thread_scope Sco, class CompletionF>
  __host__ __device__ void init(barrier<Sco,CompletionF>* bar, std::ptrdiff_t expected, CompletionF completion);
}
*/
```

- Expects: `*bar` is trivially initialized.
- Effects: equivalent to initializing `*bar` with a constructor.

In the `device::` namespace, a `__device__` free function is available that
  provides direct access to the underlying PTX state of a `barrier` object, if
  its scope is `thread_scope_block` and it is allocated in shared memory.

```c++
namespace cuda { namespace device {
  __device__ std::uint64_t* barrier_native_handle(
    barrier<thread_scope_block>& b);
}}
```

- Expects: `b` is in `__shared__` memory.
- Returns: a pointer to the PTX "mbarrier" subobject of the `barrier` object.

For example:

```c++
auto ptr = barrier_native_handle(b);

asm volatile (
    "mbarrier.arrive.b64 _, [%0];"
    :: "l"(ptr)
    : "memory");
// equivalent to: (void)b.arrive();
```

## Restrictions

An object of type `barrier` shall not be accessed concurrently by CPU and GPU
  threads unless:
- it is in unified memory and the [`concurrentManagedAccess` property] is 1, or
- it is in CPU memory and the [`hostNativeAtomicSupported` property] is 1.

Note, for objects of scopes other than `thread_scope_system` this is a
  data-race, and thefore also prohibited regardless of memory characteristics.

Under CUDA Compute Capability 8 (Ampere) or above, when an object of type
  `barrier<thread_scope_block>` is placed in `__shared__` memory, the member
  function `arrive` performs a reduction of the arrival count among
  [coalesced threads] followed by the arrival operation in one thread.
Programs shall ensure that this transformation would not introduce errors, for
  example relative to the requirements of [thread.barrier.class paragraph 12]
  of ISO/IEC IS 14882 (the C++ Standard).

Under CUDA Compute Capability 6 (Pascal) or prior, an object of type `barrier`
  may not be used.

## Implementation-Defined Behavior

For each [thread scope] `S` and completion function `F`, the value of
  `barrier<S, F>::max()` is as follows:

|Thread Scope `S`        |Completion Function `F` |`barrier<S, F>::max()`            |
|------------------------|------------------------|----------------------------------|
|`thread_scope_block`    |Default or user-provided|`(1 << 20) - 1`                   |
|Not `thread_scope_block`|Default                 |`numeric_limits<int32_t>::max()`  |
|Not `thread_scope_block`|User-provided           |`numeric_limits<ptrdiff_t>::max()`|


[thread scope]: ./thread_scopes.md

[thread.barrier.class paragraph 12]: https://eel.is/c++draft/thread.barrier.class#12

[coalesced threads]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#coalesced-group-cg

[`concurrentManagedAccess` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_116f9619ccc85e93bc456b8c69c80e78b
[`hostNativeAtomicSupported` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f
