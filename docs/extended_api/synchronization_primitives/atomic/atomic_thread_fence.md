---
grand_parent: Synchronization Primitives
parent: cuda::atomic
---

# `cuda::atomic::atomic_thread_fence`

Defined in header `<cuda/atomic>`:

```cuda
__host__ __device__
void cuda::atomic_thread_fence(cuda::std::memory_order order,
                               cuda::thread_scope scope = cuda::thread_scope_system);
```

Establishes memory synchronization ordering of non-atomic and relaxed atomic
  accesses, as instructed by `order`, for all threads within `scope` without an
  associated atomic operation.
It has the same semantics as [`cuda::std::atomic_thread_fence`].

## Example

```cuda
#include <cuda/atomic>

__global__ void example_kernel(int* data) {
  *data = 42;
  cuda::atomic_thread_fence(cuda::std::memory_order_release,
                            cuda::thread_scope_device);
}
```


[See it on Godbolt](https://godbolt.org/z/nfcoTW1Kz){: .btn }

[`cuda::std::atomic_thread_fence`]: https://en.cppreference.com/w/cpp/atomic/atomic_thread_fence
