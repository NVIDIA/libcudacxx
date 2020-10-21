---
grand_parent: Extended API
parent: Synchronization library
---

# cuda::**pipeline_shared_state**

Defined in header [`<cuda/pipeline>`](../headers/pipeline.md)

```c++
template<cuda::thread_scope Scope, uint8_t StagesCount>
class pipeline_shared_state;
```

The class template `cuda::pipeline_shared_state` is a storage type used to coordinate the threads participating in a `cuda::pipeline`.

## Template parameters

| Scope       | A [`cuda::thread_scope`](../../api/synchronization_library/thread_scopes.md) denoting a scope including all the threads participating in the `cuda::pipeline`. `Scope` cannot be `thread_scope_thread`.|
| StagesCount | The number of stages for the _pipeline_.                                                                                                                                                               |

## Member functions

| [(constructor)](./pipeline_shared_state/constructor.md) | constructs a `pipeline_shared_state`      |
| [(destructor)](./pipeline_shared_state/destructor.md)   | destroys the `pipeline_shared_state`      |
| operator= [deleted]                                     | `pipeline_shared_state` is not assignable |

## Example

```c++
#include <cuda/pipeline>

__global__ void example_kernel(char * device_buffer, char * sysmem_buffer)
{
    // Allocate a 2 stage block scoped shared state in shared memory
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, 2> pss_1;

    // Allocate a 2 stage block scoped shared state in device memory
    auto * pss_2 = new cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, 2>;

    // Construct a 2 stage device scoped shared state in device memory
    auto * pss_3 = new(device_buffer) cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_device, 2>;

    // Construct a 2 stage system scoped shared state in system memory
    auto * pss_4 = new(sysmem_buffer) cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_system, 2>;
}
```
