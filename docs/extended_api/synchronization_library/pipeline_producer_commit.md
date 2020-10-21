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

## Example

```c++
TODO
```
