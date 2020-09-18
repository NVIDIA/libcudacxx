# libcu++: The C++ Standard Library for Your Entire System

<table><tr>
<th><b><a href="https://github.com/nvidia/libcudacxx/tree/main/samples">Examples</a></b></th>
<th><b><a href="https://nvidia.github.io/libcudacxx">Documentation</a></b></th>
</tr></table>

**libcu++ is the C++ Standard Library for your entire system.**
It provides a heterogeneous implementation of the C++ Standard Library that can
  be used in and between CPU and GPU code.

If you know how to use your C++ Standard Library, then you know how to use
  libcu++.
All you have to do is add `cuda/std/` to the start of your Standard Library
  includes and `cuda::` before any uses of `std::`:

```c++
#include <cuda/std/atomic>
cuda::std::atomic<int> x;
```

libcu++ is an open source project; it is available on [GitHub] and included in
  the NVIDIA HPC SDK and CUDA Toolkit.
No additional installation or compiler flags are needed.

## `cuda::` and `cuda::std::`

When used with NVCC, libcu++ facilities live in their own header hierarchy and
  namespace with the same structure as, but distinct from, the host compiler's
  Standard Library:

* `std::`/`<*>`: Your host compiler's Standard Library that works in
      `__host__` code only.
    When using NVCC, libcu++ does not replace or interfere with host compiler's
      Standard Library.
* `cuda::std::`/`<cuda/std/*>`: Strictly conforming implementations of
      facilities from the Standard Library that work in `__host__ __device__`
      code.
* `cuda::`/`<cuda/*>`: Conforming extensions to the Standard Library that
      work in `__host__ __device__` code.
* `cuda::device`/`<cuda/device/*>`: Conforming extensions to the Standard
      Library that work only in `__device__` code.

```c++
// Standard C++, __host__ only.
#include <atomic>
std::atomic<int> x;

// CUDA C++, __host__ __device__.
// Strictly conforming to the C++ Standard.
#include <cuda/std/atomic>
cuda::std::atomic<int> x;

// CUDA C++, __host__ __device__.
// Conforming extensions to the C++ Standard.
#include <cuda/std/atomic>
cuda::atomic<int, cuda::thread_scope_block> x;
```

## libcu++ is Heterogeneous

libcu++ work across your entire codebase, both in and across host and device
  code.
libcu++ is a C++ Standard Library for your entire system, not just
Everything in `cuda::` is `__host__ __device__`.

libcu++ facilities are designed to be passed between host and device code.
Unless otherwise noted, any libcu++ object which is copyable or movable can be
  copied or moved between host and device code.

Synchronization objects work across host and device code, and can be used to
  synchronize between host and device threads.
However, there are some restrictions to be aware of; please see the
  [synchronization library section] for more details.

### `cuda::device::`

A small number of libcu++ facilities only work in device code, usually because
  there is no sensible implementation in host code.

Such facilities live in `cuda::device::`.

## `cuda::` is Incremental

libcu++ delivers a high-priority subset of the C++ Standard Library today, and
  each release increases the feature set.
But it is a subset; not everything is available today.
The [API section] lists the facilities available and the releases they were
  first introduced in.

## Licensing

libcu++ is an open source project developed on [GitHub].
It is NVIDIA's variant of [LLVM's libc++].
libcu++ is distributed under the [Apache License v2.0 with LLVM Exceptions].

## Conformance

libcu++ aims to be a conforming implementation of the C++ Standard,
  [ISO/IEC IS 14882], Clause 16 through 32.


[GitHub]: https://github.com/nvidia/libcudacxx

[API section]: https://nvidia.github.io/libcudacxx/api.html
[synchronization library section]: https://nvidia.github.io/libcudacxx/api/synchronization_library.html

[documentation]: https://nvidia.github.io/libcudacxx

[LLVM's libc++]: https://libcxx.llvm.org
[Apache License v2.0 with LLVM Exceptions]: https://llvm.org/LICENSE.txt

[ISO/IEC IS 14882]: https://eel.is/c++draft

