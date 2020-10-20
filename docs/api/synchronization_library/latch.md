---
grand_parent: API
parent: Synchronization Library
nav_order: 3
---

# `<cuda/std/latch>`, `<cuda/latch>`

## Extensions

The class template `latch` takes an additional [thread scope] argument,
  defaulted to `thread_scope_system`.

```c++
// This latch is suitable for all threads in the system.
cuda::latch<cuda::thread_scope_system> a;

// These latches has the same type as the previous one (`a`).
cuda::latch<> ba;
cuda::std::latch<> bb;

// This latch is suitable for all threads in the same thread block.
cuda::latch<cuda::thread_scope_block> c;
```

## Restrictions

An object of type `latch` shall not be accessed concurrently by CPU and GPU
  threads unless:
- it is in unified memory and the [`concurrentManagedAccess` property] is 1, or
- it is in CPU memory and the [`hostNativeAtomicSupported` property] is 1.

Note, for objects of scopes other than `thread_scope_system` this is a
  data-race, and thefore also prohibited regardless of memory characteristics.

Under CUDA Compute Capability 6 (Pascal) or prior, an object of type `latch`
  may not be used.

## Implementation-Defined Behavior

For each [thread scope] `S`, the value of `latch<S>::max()` is as follows:

|Thread Scope `S`|`latch<S>::max()`                 |
|----------------|----------------------------------|
|Any             |`numeric_limits<ptrdiff_t>::max()`|

Objects in namespace `cuda::std::` have the same behavior as corresponding
  objects in namespace `cuda::` when instantiated with a scope of
  `cuda::thread_scope_system`.


[thread scope]: ./thread_scopes.md

[`concurrentManagedAccess` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_116f9619ccc85e93bc456b8c69c80e78b
[`hostNativeAtomicSupported` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f
