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

Blocks the current thread until all operations committed to _pipelines stages_ sequenced before the `Prior` last one complete. All stages up to `Prior` (excluded)
are implicitly released.

## Template parameters

| Prior | The Nth latest stage to wait for |

## Parameters

| pipeline | the thread-scoped `cuda::pipeline` object to wait on |

## Example

```c++
TODO
```
