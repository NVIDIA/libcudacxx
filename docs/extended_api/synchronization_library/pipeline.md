---
grand_parent: Extended API
parent: Synchronization library
---

# cuda::**pipeline**

Defined in header [`<cuda/pipeline>`](../headers/pipeline.md)

```c++
template<cuda::thread_scope Scope>
class pipeline;
```

The class template `cuda::pipeline` provides a coordination mechanism allowing to pipeline multiple operations in a sequence of stages.

A thread interacts with a _pipeline stage_ using the following pattern:
1. Acquire the pipeline stage
2. Commit some operations to the stage
3. Wait for the previously committed operations to complete
4. Release the pipeline stage

For thread scopes other than `thread_scope_thread`, a [`pipeline_shared_state`](./pipeline_shared_state.md) is required to coordinate the participating threads.

_Pipelines_ can be either _unified_ or _partitioned_.
In a _unified pipeline_, all the participating threads are both producers and consumers.
In a _partitioned pipeline_, each participating thread is either a producer or (exclusive) a consumer.

## Template parameters

### Scope

A [`cuda::thread_scope`](../../api/synchronization_library/thread_scopes.md) denoting a scope including all the threads participating in the _pipeline_.

## Member functions

| (constructor) [deleted]                            | `pipeline` is not constructible                                                                                                                  |
| [(destructor)](./pipeline/destructor.md)           | destroys the `pipeline`                                                                                                                          |
| operator= [deleted]                                | `pipeline` is not assignable                                                                                                                     |
| [producer_acquire](./pipeline/producer_acquire.md) | blocks the current thread until the next _pipeline stage_ is available                                                                           |
| [producer_commit](./pipeline/producer_commit.md)   | commits operations previously issued by the current thread to the current _pipeline stage_                                                       |
| [consumer_wait](./pipeline/consumer_wait.md)       | blocks the current thread until all operations committed to the current _pipeline stage_ complete                                                |
| [consumer_wait_for](./pipeline/consumer_wait.md)   | blocks the current thread until all operations committed to the current _pipeline stage_ complete or after the specified timeout duration        |
| [consumer_wait_until](./pipeline/consumer_wait.md) | blocks the current thread until all operations committed to the current _pipeline stage_ complete or until specified time point has been reached |
| [consumer_release](./pipeline/consumer_release.md) | release the current _pipeline stage_                                                                                                             |
| [quit](./pipeline/quit.md)                         | quits current thread's participation in the _pipeline_                                                                                           |

## Notes

A thread role cannot change during the lifetime of the pipeline object.

## Example

```c++
TODO
```
