---
grand_parent: Extended API
parent: Barriers
---

# `cuda::barrier::wait_parity` and `cuda::barrier::try_wait_parity`

Defined in header `<cuda/std/barrier>`:

```cuda
__host__ __device__ void cuda::std::barrier::wait_parity(bool phase);
__host__ __device__ bool cuda::std::barrier::try_wait_parity(bool phase);
```

`barrier::wait_parity` stalls execution while the barrier is not at the specified parity.
`barrier::try_wait_parity` returns true if the parity of the barrier matches the given parity.

## Return Value

`barrier::try_wait_parity` returns a boolean representing whether the barrier is at the given phase

<!-- TODO: Create an example when trunk is live on godbolt
## Example

```cuda
#include <cuda/barrier>

__global__ void example_kernel(cuda::barrier<cuda::thread_scope_block>& bar) {
  bar.wait_parity(false);
}
```
-->

[See it on Godbolt](https://godbolt.org/z/dr4798Y76){: .btn }


[`cuda::thread_scope`]: ./thread_scopes.md

[thread.barrier.class paragraph 12]: https://eel.is/c++draft/thread.barrier.class#12

[coalesced threads]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#coalesced-group-cg

[`concurrentManagedAccess` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_116f9619ccc85e93bc456b8c69c80e78b
[`hostNativeAtomicSupported` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f

