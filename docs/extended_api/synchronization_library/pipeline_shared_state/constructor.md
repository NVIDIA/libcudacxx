---
nav_exclude: true
---

# cuda::pipeline_shared_state\<Scope, StagesCount>::**pipeline_shared_state**

```c++
pipeline_shared_state();                                       // (1)
pipeline_shared_state(const pipeline_shared_state &) = delete; // (2)
pipeline_shared_state(pipeline_shared_state &&) = delete;      // (3)
```

1. Constructs the pipeline shared state.
2. Copy constructor is deleted.
3. Move constructor is deleted.

## Notes

Static declaration of `pipeline_shared_state` within device code currently emits the following warning:

```
warning: dynamic initialization is not supported for a function-scope static __shared__ variable within a __device__/__global__ function
```

It can be silenced using `#pragma diag_suppress static_var_with_dynamic_init`.

## Example

```c++
#include <cuda/pipeline>

#pragma diag_suppress static_var_with_dynamic_init

__global__ void example_kernel()
{
    __shared__ cuda::pipeline_shared_state<cuda::thread, 2> shared_state;
}
```
