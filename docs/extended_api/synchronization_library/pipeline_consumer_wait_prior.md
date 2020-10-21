---
grand_parent: Extended API
parent: Synchronization library
---

# cuda::**pipeline_consumer_wait_prior**

Defined in header [`<cuda/pipeline>`](../headers/pipeline.md)

```c++
template<uint8_t Prior>
void pipeline_consumer_wait_prior(pipeline<thread_scope_thread> & pipeline);
```

Let *Stage* be the pipeline stage `Prior` stages before the current one (counting the current one). Blocks the current
thread until all operations committed to _pipeline stages_ up to *Stage* complete. All stages up to *Stage* (exclusive)
are implicitly released.

## Template parameters

| Prior | The index of the pipeline stage *Stage* (_see above_) counting up from the current one. The index of the current stage is `0`. |

## Parameters

| pipeline | The thread-scoped `cuda::pipeline` object to wait on |

## Notes

If the pipeline is in a _quitted state_ (see [`pipeline::quit`](./pipeline/quit.md)), the behavior is undefined.

## Example

```c++
#include <cuda/pipeline>

__global__ void example_kernel(uint64_t * global, size_t element_count)
{
    extern __shared__ uint64_t shared[];

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    for (size_t i = 0; i < element_count; ++i) {
        pipe.producer_acquire();
        cuda::memcpy_async(shared + i, global + i, sizeof(*global), pipe);
        pipe.producer_commit();
    }

    // Wait for operations committed in all stages but the last one
    cuda::pipeline_consumer_wait_prior<1>(pipe);
    pipe.consumer_release();

    // Wait for operations committed in all stages
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    pipe.consumer_release();
}
```

[See it on Godbolt](https://godbolt.org/z/j83v3G){: .btn }
