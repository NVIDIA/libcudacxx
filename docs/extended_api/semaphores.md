---
parent: Extended API
nav_order: 6
---

# Semaphores

Defined in header `<cuda/semaphore>`:

```cuda
namespace cuda {

template <cuda::thread_scope Scope,
          cuda::std::ptrdiff_t LeastMaxValue = /* implementation-defined */>
class counting_semaphore;

template <cuda::thread_scope Scope>
using binary_semaphore = std::counting_semaphore<Scope, 1>;

}
```

The class templates `cuda::counting_semaphore` and `cuda::binary_semaphore`
  are extended forms of `cuda::std::counting_semaphore` and
  `cuda::std::binary_semaphore` take an additional [`cuda::thread_scope`]
  argument.
`cuda::counting_semaphore` has the same interface and semantics as
  [`cuda::std::counting_semaphore`].
`cuda::binary_semaphore` has the same interface and semantics as
  [`cuda::std::binary_semaphore`], but `cuda::binary_semaphore` is a class
  template.

## Concurrency Restrictions

An object of type `cuda::counting_semaphore`, `cuda::std::counting_semaphore`,
  `cuda::binary_semaphore`, or `cuda::std::binary_semaphore` shall not be
  accessed concurrently by CPU and GPU threads unless:
- it is in unified memory and the [`concurrentManagedAccess` property] is 1, or
- it is in CPU memory and the [`hostNativeAtomicSupported` property] is 1.

Note, for objects of scopes other than `cuda::thread_scope_system` this is a
  data-race, and thefore also prohibited regardless of memory characteristics.

Under CUDA Compute Capability 6 (Pascal) or prior, an object of type
  `cuda::counting_semaphore`, `cuda::std::semaphore`, `cuda::binary_semaphore`
  or `cuda::std::binary_semaphore` may not be used.

## Implementation-Defined Behavior

For each [`cuda::thread_scope`] `S`, `cuda::binary_semaphore<S>::max()` is as
  follows:

| [`cuda::thread_scope`] `S` | `cuda::binary_semaphore<S>::max()` |
|----------------------------|------------------------------------|
| Any                        | `1`                                |

For each [`cuda::thread_scope`] `S` and least maximum value `V`,
  `counting_semaphore<S,V>::max()` is as follows:

| [`cuda::thread_scope`] `S` | Least Maximum Value `V` | `cuda::counting_semaphore<S,V>::max()`                   |
|----------------------------|-------------------------|----------------------------------------------------------|
| Any                        | Any                     | `cuda::std::numeric_limits<cuda::std::ptrdiff_t>::max()` |

## Example

```cuda
#include <cuda/semaphore>

__global__ void example_kernel() {
  // These semaphores are suitable for all threads in the system.
  cuda::binary_semaphore<cuda::thread_scope_system> a0;
  cuda::counting_semaphore<cuda::thread_scope_system> a1;

  // These semaphores have the same types as the previous ones.
  cuda::std::binary_semaphore b0;
  cuda::std::counting_semaphore<> b1;

  // These semaphores are suitable for all threads on the current processor (e.g. GPU).
  cuda::binary_semaphore<cuda::thread_scope_device> c0;
  cuda::counting_semaphore<cuda::thread_scope_device> c1;

  // These semaphores are suitable for all threads in the same thread block.
  cuda::binary_semaphore<cuda::thread_scope_block> d0;
  cuda::counting_semaphore<cuda::thread_scope_block> d1;
}
```

[See it on Godbolt](https://godbolt.org/z/8oqPj9){: .btn }


[`cuda::thread_scope`]: ./thread_scopes.md

[`cuda::std::binary_semaphore`]: https://en.cppreference.com/w/cpp/thread/binary_semaphore
[`cuda::std::counting_semaphore`]: https://en.cppreference.com/w/cpp/thread/counting_semaphore

[`concurrentManagedAccess` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_116f9619ccc85e93bc456b8c69c80e78b
[`hostNativeAtomicSupported` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f
