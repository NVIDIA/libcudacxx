---
grand_parent: Extended API
parent: Synchronization library
---

# cuda::**pipeline_role**

Defined in header [`<cuda/pipeline>`](../headers/pipeline.md)

```c++
enum class pipeline_role : /* unspecified */ {
    producer,
    consumer
};
```
`cuda::pipeline_role` specifies the role of a particular thread in a partitioned producer/consumer pipeline.

## Constants

| producer | a producer thread generates data (e.g. by issuing [`memcpy_async`](../asynchronous_operations_library/memcpy_async.md) operations)                           |
| consumer | a consumer thread consumes data (e.g. by waiting for previously [`memcpy_async`](../asynchronous_operations_library/memcpy_async.md) operations to complete) |

## Example

See [cuda::make_pipeline](./make_pipeline.md#example).
