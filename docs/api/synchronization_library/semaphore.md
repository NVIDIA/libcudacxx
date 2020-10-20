---
grand_parent: API
parent: Synchronization Library
nav_order: 4
---

# `<cuda/std/semaphore>`, `<cuda/semaphore>`

## Extensions

The class `binary_semaphore` is a class template in this namespace.

The class templates `binary_semaphore` and `counting_semaphore` take an
  additional [thread scope] argument, defaulted to `thread_scope_system`.

```c++
// These semaphores are suitable for all threads in the system.
cuda::binary_semaphore<cuda::thread_scope_system> a0;
cuda::counting_semaphore<cuda::thread_scope_system> a1;

// These semaphores have the same types as the previous ones (`a0` and `a1` respectively).
cuda::binary_semaphore<> b0a;
cuda::std::binary_semaphore b0b;
cuda::counting_semaphore<> b1a;
cuda::std::counting_semaphore<> b1b;

// These semaphores are suitable for all threads in the same thread block.
cuda::binary_semaphore<cuda::thread_scope_block> c0;
cuda::counting_semaphore<cuda::thread_scope_block> c1;
```

## Restrictions

An object of type `counting_semaphore` or `binary_semaphore` shall not be
  accessed concurrently by CPU and GPU threads unless:
- it is in unified memory and the [`concurrentManagedAccess` property] is 1, or
- it is in CPU memory and the [`hostNativeAtomicSupported` property] is 1.

Note, for objects of scopes other than `thread_scope_system` this is a
  data-race, and thefore also prohibited regardless of memory characteristics.

Under CUDA Compute Capability 6 (Pascal) or prior, an object of type
  `counting_semaphore` or `binary_semaphore` may not be used.

## Implementation-Defined Behavior

For each [thread scope] `S`, `binary_semaphore<S>::max()` is as follows:

|Thread Scope `S`|`binary_semaphore<S>::max()`|
|----------------|----------------------------|
|Any             |`1`                         |

For each [thread scope] `S` and least maximum value `V`,
  `counting_semaphore<S,V>::max()` is as follows:

|Thread Scope `S`|Least Maximum Value `V`|`counting_semaphore<S,V>::max()`  |
|----------------|-----------------------|----------------------------------|
|Any             |Any                    |`numeric_limits<ptrdiff_t>::max()`|


[thread scope]: ./thread_scopes.md

[`concurrentManagedAccess` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_116f9619ccc85e93bc456b8c69c80e78b
[`hostNativeAtomicSupported` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f
