---
grand_parent: Extended API
parent: Functional
---

:warning: **The libcudacxx repository has been archived and is now part of the unified [nvidia/cccl repository](https://github.com/nvidia/cccl). See the [announcement here](https://github.com/NVIDIA/cccl/discussions/520) for more information. Please visit the new repository for the latest updates. The new documentation can be found [here](https://nvidia.github.io/cccl/libcudacxx/).** :warning:

# `cuda::proclaim_return_type`

Defined in the header `<cuda/functional>`:

```cuda
template <class Ret, class Fn>
__host__ __device__
unspecified<Ret, Fn> proclaim_return_type(Fn&& fn) {
  return unspecified<Ret, Fn>{fn};
}
```

`cuda::proclaim_return_type` creates a forwarding call wrapper that uses
`Ret` as a return type. The wrapper is useful in the case of extended device
lambdas since an attempt to determine the return type of their `operator()` 
function may work incorrectly in host code.

## Template Parameters

| `Ret` | Return type that's being proclaimed       |
| `Fn`  | Callable object type that's being wrapped |

## Parameters

| `fn`  | Callable object that's being wrapped |

## Example

```cuda
#include <cuda/functional>

template <class T, class Fn>
__global__ void example_kernel(T *out, Fn fn) {
  *out = fn();
}

__host__ void example() {
  auto fn = cuda::proclaim_return_type<char>([] __device__ () { return 'd'; });
  using rt = cuda::std::invoke_result_t<decltype(fn)>;

  rt* out {}; 
  cudaMalloc(&out, sizeof(rt));

  example_kernel<<<1, 1>>>(out, fn);

  // ...
}

```

