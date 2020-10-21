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

| Group | a type satisfying the [_group concept_](http://LINK-TODO) |

## Parameters

| group          | the group of threads                                                                                                                   |
| shared_state   | an object of type [`cuda::pipeline_shared_state<Scope>`](./pipeline_shared_state.md) with `Scope` including all the threads in `group` |
| producer_count | the number of _producer threads_ in the pipeline                                                                                       |
| role           | the role of the current thread in the pipeline                                                                                         |

## Return value

A thread-local `cuda::pipeline` object.

## Example

```c++
TODO
```
