---
grand_parent: Extended API
parent: Synchronization library
---

# cuda::**make_pipeline**

Defined in header [`<cuda/pipeline>`](../headers/pipeline.md)

```c++
pipeline<thread_scope_thread> make_pipeline();                                                                                       // (1)

template<class Group, thread_scope Scope, uint8_t StagesCount>
pipeline<Scope> make_pipeline(const Group & group, pipeline_shared_state<Scope, StagesCount> * shared_state);                        // (2)

template<class Group, thread_scope Scope, uint8_t StagesCount>
pipeline<Scope> make_pipeline(const Group & group, pipeline_shared_state<Scope, StagesCount> * shared_state, size_t producer_count); // (3)

template<class Group, thread_scope Scope, uint8_t StagesCount>
pipeline<Scope> make_pipeline(const Group & group, pipeline_shared_state<Scope, StagesCount> * shared_state, pipeline_role role);    // (4)
```

1. Creates a _unified pipeline_ such that the calling thread is the only participating thread and performs both producer and consumer actions.
2. Creates a _unified pipeline_ such that all the threads in `group` are performing both producer and consumer actions.
3. Creates a _partitioned pipeline_ such that `producer_threads` number of threads in `group` are performing producer actions while the others
   are performing consumer actions. 
4. Creates a _partitioned pipeline_ where each thread's role is explicitly specified.

All threads in `group` acquire collective ownership of the `shared_state` storage.

`make_pipeline` must be invoked by every threads in `group` such that `group::sync` may be invoked.

`shared_state` and `producer_count` must be uniform across all threads in `group`, else the behavior is undefined.

`producer_count` must be strictly inferior to `group::size`, else the behavior is undefined.

## Template parameters

| Group | a type satisfying the [_Group concept_](../concepts/group.md) |

## Parameters

| group          | the group of threads                                                                                                                                |
| shared_state   | a pointer to an object of type [`cuda::pipeline_shared_state<Scope>`](./pipeline_shared_state.md) with `Scope` including all the threads in `group` |
| producer_count | the number of _producer threads_ in the pipeline                                                                                                    |
| role           | the role of the current thread in the pipeline                                                                                                      |

## Return value

A `cuda::pipeline` object.

## Example

```c++
#include <cuda/pipeline>
#include <cooperative_groups.h>

// Disables `pipeline_shared_state` initialization warning
#pragma diag_suppress static_var_with_dynamic_init

__global__ void example_kernel()
{
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pss_1;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pss_2;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pss_3;

    auto group = cooperative_groups::this_thread_block();

    // Create a thread scoped pipeline
    cuda::pipeline<cuda::thread_scope_thread> p_0 = cuda::make_pipeline();

    // Create a unified block-scoped pipeline
    cuda::pipeline<cuda::thread_scope_block> p_1 = cuda::make_pipeline(group, &pss_1);

    // Create a partitioned block-scoped pipeline where half the threads are producers
    size_t producer_count = group.size() / 2;
    cuda::pipeline<cuda::thread_scope_block> p_2 = cuda::make_pipeline(group, &pss_2, producer_count);

    // Create a partitioned block-scoped pipeline where all threads with an even thread_rank are producers
    auto thread_role = (group.thread_rank() % 2) ? cuda::pipeline_role::producer : cuda::pipeline_role::consumer;
    cuda::pipeline<cuda::thread_scope_block> p_3 = cuda::make_pipeline(group, &pss_3, thread_role);
}
```

[See it on Godbolt](https://godbolt.org/z/Y1zv5G){: .btn }
