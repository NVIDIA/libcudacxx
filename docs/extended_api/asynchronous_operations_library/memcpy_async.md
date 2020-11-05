---
grand_parent: Extended API
parent: Asynchronous operations library
---

# cuda::**memcpy_async**

Defined in header [`<cuda/barrier>`](../../api/synchronization_library/barrier.md)

```c++
template<typename Size, thread_scope Scope>
void memcpy_async(void * destination, void const * source, Size size, barrier<Scope> & barrier);                        // (1)

template<typename Group, typename Size, thread_scope Scope>
void memcpy_async(Group const & group, void * destination, void const * source, Size size, barrier<Scope> & barrier);   // (2)
```

Defined in header [`<cuda/pipeline>`](../headers/pipeline.md)

```c++
template<typename Size, thread_scope Scope>
void memcpy_async(void * destination, void const * source, Size size, pipeline<Scope> & pipeline);                      // (3)

template<typename Group, typename Size, thread_scope Scope>
void memcpy_async(Group const & group, void * destination, void const * source, Size size, pipeline<Scope> & pipeline); // (4)
```

Asynchronously copies `size` bytes from the memory location pointed to by `source` to the memory location pointed to by `destination`.
Both objects are reinterpreted as arrays of `unsigned char`.

`cuda::memcpy_async` have similar constraints to [`std::memcpy`](https://en.cppreference.com/w/cpp/string/byte/memcpy), namely:
* If the objects overlap, the behavior is undefined.
* If either `destination` or `source` is an invalid or null pointer, the behavior is undefined (even if `count` is zero).
* If the objects are potentially-overlapping or not [`TriviallyCopyable`](https://en.cppreference.com/w/cpp/named_req/TriviallyCopyable),
  the behavior is undefined.

1. Binds the asynchronous copy completion to `barrier` and issues the copy in the current thread.
2. Binds the asynchronous copy completion to `barrier` and cooperatively issues the copy across all threads in `group`.
3. Binds the asynchronous copy completion to `pipeline` and issues the copy in the current thread
4. Binds the asynchronous copy completion to `pipeline` and cooperatively issues the copy across all threads in `group`.

## Template parameters

| Group | a type satisfying the [_group concept_](http://LINK-TODO)                                                 |
| Size  | [`size_t`](https://en.cppreference.com/w/c/types/size_t) or [`cuda::aligned_size_t`](./aligned_size_t.md) |

## Parameters

| group       | the group of threads                                    |
| destination | pointer to the memory location to copy to               |
| source      | pointer to the memory location to copy from             |
| size        | the number of bytes to copy                             |
| barrier     | the barrier object used to wait on the copy completion  |
| pipeline    | the pipeline object used to wait on the copy completion |

## Example

CUDA kernels often first copy data from global to shared memory, to then perform a computation using that shared memory data ([live](https://cuda.godbolt.org/z/34PMMe)):

```c++
#include <cooperative_groups.h>
__device__ void compute(float*);

__global__ void kernel(float* global, size_t subset_count) {
  extern __shared__ float shared[];
  auto group = cooperative_groups::this_thread_block();
  size_t shared_idx = group.thread_rank();

  for (size_t subset = 0; subset < subset_count; ++subset) {
    size_t global_idx = subset * group.size() + group.thread_rank();
    shared[shared_idx] = global[global_idx];  // Fetch from global to shared
    group.sync();                             // Wait for copy to complete
    compute(shared);                          // Compute
    group.sync();                             // Wait for compute to complete
  }
}
```

With `cuda::memcpy_async` we can overlap the global to shared memory copies of the next batch, with the computation on the current batch, by using a two-stage pipeline as follows ([live](https://cuda.godbolt.org/z/GMGe8P)):

```c++
#include <cooperative_groups.h>
#include <cuda/pipeline>
__device__ void compute(float*);

__global__ void kernel(float* global, size_t subset_count) {
  auto group = cooperative_groups::this_thread_block();

  constexpr unsigned stages_count = 2;
  extern __shared__ float s[];
  // Two batches must fit in shared memory, pointers to each batch:
  float * shared[stages_count] = { s, s + group.size() };
 
  // Allocate shared storage for a two-stage cuda::pipeline:
  __shared__ cuda::pipeline_shared_state<
    cuda::thread_scope::thread_scope_block,
    stages_count
  > shared_state;
  auto pipeline = cuda::make_pipeline(group, &shared_state);
 
  for (size_t subset = 0, fetch = 0; subset < subset_count; ++subset) {
    size_t stage_idx = fetch % 2; 
    // Fetche up to `stages_count` subsets ahead:
    for (; fetch < subset_count && fetch < (subset + stages_count); ++fetch ) {
      // Collectively acquire the pipeline head stage from all producer threads:
      pipeline.producer_acquire();
      size_t global_idx = (subset + 1) * group.size();
      cuda::memcpy_async(  // Submit async copies to the pipeline's head stage
        group,
        shared[stage_idx],
        &global[global_idx],
        sizeof(float) * group.size(),
        pipeline
      );
      // Collectively commit (advance) the pipeline's head stage
      pipeline.producer_commit(); 
    }
    // Collectively wait for the operations commited to the
    // current `subset` stage to complete:
    pipeline.consumer_wait(); 
    compute(shared[stage_idx]);
    
    // Collectively release the stage resources
    pipeline.consumer_release();
  }
}
```
