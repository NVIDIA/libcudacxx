---
grand_parent: Extended API
parent: Synchronization library
---

# cuda::**pipeline_producer_commit**

Defined in header [`<cuda/pipeline>`](../headers/pipeline.md)

```c++
template<thread_scope Scope>
void pipeline_producer_commit(pipeline<thread_scope_thread> & pipeline, barrier<Scope> & barrier);
```

Binds operations previously issued by the current thread to the named `barrier` such that a `barrier::arrive` is performed on completion. The bind operation implicitly increments the barrier's current phase to account for the subsequent `barrier::arrive`, resulting in a net change of 0.

## Parameters

| pipeline | the thread-scoped `cuda::pipeline` object to wait on |
| barrier  | the barrier to arrive on                             |

## Notes

If the pipeline is in a _quitted state_ (see [`pipeline::quit`](./pipeline/quit.md)), the behavior is undefined.

## Example

```c++
#include <cuda/pipeline>

// Disables `barrier` initialization warning
#pragma diag_suppress static_var_with_dynamic_init

__global__ void example_kernel(uint64_t * global, size_t element_count)
{
    extern __shared__ uint64_t shared[];
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

    init(&barrier, 1);
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    pipe.producer_acquire();
    for (size_t i = 0; i < element_count; ++i) {
        cuda::memcpy_async(shared + i, global + i, sizeof(*global), pipe);
    }
    pipeline_producer_commit(pipe, barrier);
    barrier.arrive_and_wait();
    pipe.consumer_release();
}
```

[See it on Godbolt](https://godbolt.org/z/x5n8zY){: .btn }
