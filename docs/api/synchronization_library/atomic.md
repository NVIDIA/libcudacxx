---
grand_parent: API
parent: Synchronization Library
nav_order: 1
---

# `<cuda/std/atomic>`, `<cuda/atomic>`

## Extensions

The class template `atomic` takes an additional [thread scope] argument,
  defaulted to `thread_scope_system`.

```c++
// This atomic is suitable for all threads in the system.
cuda::atomic<int, cuda::thread_scope_system> a;

// This atomic has the same type as the previous one (`a`).
cuda::atomic<int> b;

// This atomic is suitable for threads in the same thread block.
cuda::atomic<int, cuda::thread_scope_block> c;
```

The `atomic` class template specializations for integral and pointer types are
extended with members `fetch_min` and `fetch_max`.
These conform to the requirements in section [atomics.types.int] and
  [atomics.types.pointer] of ISO/IEC IS 14882 (the C++ Standard),
  with keys _min_ and _max_, and operations `min` and `max`, respectively.

Note that conformance to these requirements include implicit conversions to
  unsigned types, if applicable.

```c++
cuda::atomic<int> a(1);
auto x = a.fetch_min(0); // Operates as if unsigned.
auto y = a.load();
assert(x == 1 && y == 0);
```

## Restrictions

An object of type `atomic` shall not be accessed concurrently by CPU and GPU
  threads unless:
- it is in unified memory and the [`concurrentManagedAccess` property] is 1, or
- it is in CPU memory and the [`hostNativeAtomicSupported` property] is 1.

Note, for objects of scopes other than `thread_scope_system` this is a
  data-race, and thefore also prohibited regardless of memory characteristics.

Under CUDA Compute Capability 6 (Pascal), an object of type `atomic` may not be
  used:
- with automatic storage duration, or
- if `is_always_lock_free()` is `false`.

Under CUDA Compute Capability prior to 6 (Pascal), objects of type `atomic` may
not be used.

## Implementation-Defined Behavior

For each type `T` and [thread scope] `S`, the value of
  `atomic<T, S>::is_always_lock_free()` is as follows:

|Type `T`|Thread Scope `S`|`atomic<T, S>::is_always_lock_free()`|
|--------|----------------|-------------------------------------|
|Any     |Any             |`sizeof(T) <= 8`                     |


[thread scope]: ./thread_scopes.md

[atomics.types.int]: https://eel.is/c++draft/atomics.types.int
[atomics.types.pointer]: https://eel.is/c++draft/atomics.types.pointer

[`concurrentManagedAccess` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_116f9619ccc85e93bc456b8c69c80e78b
[`hostNativeAtomicSupported` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f
