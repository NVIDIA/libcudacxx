---
parent: Releases
nav_order: 0
---

# Changelog

## libcu++ 1.4.0

libcu++ 1.4.0 adds `<cuda/std/complex>`, NVCC + MSVC support for
  `<cuda/std/tuple>` and `cuda::std::pair`, and backports of C++20
  `<cuda/std/chrono>` and C++17 `<cuda/std/type_traits>` features.

Supported ABI versions: 3 (default) and 2.

## New Features

## Issues Fixed

## libcu++ 1.3.0 (CUDA Toolkit 11.2)

libcu++ 1.3.0 adds `<cuda/std/tuple>` and `cuda::std::pair`, although they are
  not supported with NVCC + MSVC.
It also adds [documentation](https://nvidia.github.io/libcudacxx).

Supported ABI versions: 3 (default) and 2.

Included in: CUDA 11.2.

### New Features

- #17: `<cuda/std/tuple>`: `cuda::std::tuple`, a fixed-size collection of
    heterogeneous values.
  Not supported with NVCC + MSVC.
- #17: `<cuda/std/utility>`: `cuda::std::pair`, a collection of two
    heterogeneous values.
  The only `<cuda/std/utility>` facilities supported are `cuda::std::pair`.
  Not supported with NVCC + MSVC.

### Other Enhancements

- [Documentation](https://nvidia.github.io/libcudacxx).

### Issues Fixed

- #21: Disable `__builtin_is_constant_evaluated` usage with NVCC in C++11 mode
    because it's broken.
- #25: Fix some declarations/definitions in `__threading_support` which have
    inconsistent qualifiers.
  Thanks to Gonzalo Brito Gadeschi for this contribution.

## libcu++ 1.2.0 (CUDA Toolkit 11.1)

libcu++ 1.2.0 adds `<cuda/pipeline>`/`cuda::pipeline`, a facility for
  coordinating `cuda::memcpy_async` operations.
This release introduces ABI version 3, which is now the default.

Supported ABI versions: 3 (default) and 2.

Included in: CUDA 11.1.

### ABI Breaking Changes

- ABI version 3 has been introduced and is now the default.
  A new ABI version was necessary to improve the performance of
    `cuda::[std::]barrier` by changing its alignment.
  Users may define `_LIBCUDACXX_CUDA_ABI_VERSION=2` before including any libcu++
    or CUDA headres to use ABI version 2, which was the default for the 1.1.0 /
    CUDA 11.0 release.
  Both ABI version 3 and ABI version 2 will be supported until the next major
    CUDA release.

### New Features

- `<cuda/pipeline>`: `cuda::pipeline`, a facility for coordinating
    `cuda::memcpy_async` operations.
- `<cuda/std/version>`: API version macros `_LIBCUDACXX_CUDA_API_VERSION`,
    `_LIBCUDACXX_CUDA_API_VERSION_MAJOR`, `_LIBCUDACXX_CUDA_API_VERSION_MINOR`,
    `_LIBCUDACXX_CUDA_API_VERSION_PATCH`.
- ABI version switching: users can define `_LIBCUDACXX_CUDA_ABI_VERSION`
    to request a particular supported ABI version.
  `_LIBCUDACXX_CUDA_ABI_VERSION_LATEST` is set to the latest ABI version, which
    is always the default.

### Other Enhancements

- `<cuda/latch>`/`<cuda/semaphore>`: `<cuda/*>` headers added for `cuda::latch`,
    `cuda::counting_semaphore`, and `cuda::binary_semaphore`.
   These features were available in prior releases, but you had to include
      `<cuda/std/latch>` and `<cuda/std/semaphore>` to access them.
- NVCC + GCC 10 support.
- NVCC + Clang 10 support.

## libcu++ 1.1.0 (CUDA Toolkit 11.0)

libcu++ 1.1.0 introduces the world's first implementation of the
  [Standard C++20 synchronization library](https://wg21.link/P1135):
  `<cuda/[std/]barrier>`, `<cuda/std/latch>`, `<cuda/std/semaphore>`,
  `cuda::[std::]atomic_flag::test`, `cuda::[std::]atomic::wait`, and
  `cuda::[std::]atomic::notify*`.
An extension for managing asynchronous local copies, `cuda::memcpy_async` is
  introduced as well.
It also adds `<cuda/std/chrono>`, `<cuda/std/ratio>`, and most of
  `<cuda/std/functional>`.

### ABI Breaking Changes

- ABI version 2 has been introduced and is now the default.
  A new ABI version was introduced because it is our policy to do so in every
    major CUDA toolkit release.
  ABI version 1 is no longer supported.

### API Breaking Changes

- Atomics on Pascal + Windows are disabled because the platform does not support
    them and on this platform the CUDA driver rejects binaries containing these
    operations.

### New Features

- `<cuda/[std/]barrier>`: C++20's `cuda::[std::]barrier`, an asynchronous thread
    coordination mechanism whose lifetime consists of a sequence of barrier
    phases, where each phase allows at most an expected number of threads to
    block until the expected number of threads arrive at the barrier.
  It is backported to C++11.
  The `cuda::barrier` variant takes an additional `cuda::thread_scope` parameter.
- `<cuda/barrier>`: `cuda::memcpy_async`, asynchronous local copies.
  This facility is NOT for transferring data between threads or transferring
    data between host and device; it is not a `cudaMemcpyAsync` replacement or
    abstraction.
  It uses `cuda::[std::]barrier`s objects to synchronize the copies.
- `<cuda/std/functional>`: common function objects, such as `cuda::std::plus`,
    `cuda::std::minus`, etc.
  `cuda::std::function`, `cuda::std::bind`, `cuda::std::hash`, and
    `cuda::std::reference_wrapper` are omitted.

### Other Enhancements

- Upgraded to a newer version of upstream libc++.
- Standalone NVRTC support.
- C++17 support.
- NVCC + GCC 9 support.
- NVCC + Clang 9 support.
- Build with warnings-as-errors.

### Issues Fixed

- Made `__cuda_memcmp` inline to fix ODR violations when compiling multiple
  translation units.

## libcu++ 1.0.0 (CUDA Toolkit 10.2)

libcu++ 1.0.0 is the first release of libcu++, the C++ Standard Library for your
  entire system.
It brings C++ atomics to CUDA: `<cuda/[std/]atomic>`.
It also introduces `<cuda/std/type_traits>`, `<cuda/std/cassert>`,
`<cuda/std/cfloat>`, `<cuda/std/cstddef>`, and `<cuda/std/cstdint>`.

### New Features

- `<cuda/[std/]atomic>`:
  - `cuda::thread_scope`: An enumeration that specifies which group of threads
    can synchronize with each other using a concurrency primitive.
  - `cuda::atomic<T, Scope>`: Scoped atomic objects.
  - `cuda::std::atomic<T>`: Atomic objects.
- `<cuda/std/type_traits>`: Type traits and metaprogramming facilities.
- `<cuda/std/cassert>`: `assert`, an error-reporting mechanism.
- `<cuda/std/cstddef>`: Builtin fundamental types.
- `<cuda/std/cstdint>`: Builtin integral types.
- `<cuda/std/cfloat>`: Builtin floating point types.

### Known Issues

- Due to circumstances beyond our control, the NVIDIA-provided Debian packages
    install libcu++ to the wrong path.
  This makes libcu++ unusable if installed from the NVIDIA-provided Debian
    packages and may interfere with the operation of your host C++ Standard
    library.

