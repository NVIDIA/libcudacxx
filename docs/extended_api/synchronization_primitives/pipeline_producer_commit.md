---
grand_parent: Extended API
parent: Synchronization Primitives
---

# `cuda::pipeline_producer_commit`

Defined in header `<cuda/pipeline>`:

```cuda
template <cuda::thread_scope Scope>
__host__ __device__
void cuda::pipeline_producer_commit(cuda::pipeline<cuda::thread_scope_thread>& pipe,
                                    cuda::barrier<Scope>& bar);
```

Binds operations previously issued by the current thread to the named
  `cuda::barrier` such that a `cuda::barrier::arrive` is performed on completion.
The bind operation implicitly increments the barrier's current phase to account
  for the subsequent `cuda::barrier::arrive`, resulting in a net change of 0.

## Parameters

| `pipe` | The thread-scoped `cuda::pipeline` object to wait on. |
| `bar`  | The `cuda::barrier` to arrive on.                     |

## Notes

If the pipeline is in a _quitted state_ (see [`cuda::pipeline::quit`]), the
  behavior is undefined.

## Example

```cuda
#include <cuda/pipeline>

// Disables `barrier` initialization warning.
#pragma diag_suppress static_var_with_dynamic_init

__global__ void
example_kernel(cuda::std::uint64_t* global, cuda::std::size_t element_count) {
  extern __shared__ cuda::std::uint64_t shared[];
  __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

  init(&barrier, 1);
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  pipe.producer_acquire();
  for (cuda::std::size_t i = 0; i < element_count; ++i)
    cuda::memcpy_async(shared + i, global + i, sizeof(*global), pipe);
  pipeline_producer_commit(pipe, barrier);
  barrier.arrive_and_wait();
  pipe.consumer_release();
}
```

[See it on Godbolt](https://godbolt.org/z/GdEc6G){: .btn }


[`cuda::pipeline::quit`]: ./pipeline/quit.md

