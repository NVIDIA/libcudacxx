# A _freestanding_ Standard C++ library for GPU programs.

Implements the eponymous subset of C++ with some exceptions.

## How to use this library

## Clone this repo

```
git clone --recurse-submodules https://github.com/ogiroux/freestanding
```

## Run the sample

On Linux, for example.

```
cd samples
./linux.sh
./books.sh
./trie
```

(_Note: you will need `curl`, and obviously CUDA with a Volta, Xavier or Turing GPU._)

## What is supported

Nothing. This repository holds a useful demo. That's all.

You may, however, enjoy creating your own demo application.

## What happens to work

Toolchains based on CUDA 9.0 and above:
1. NVCC with GCC or Clang, and the libstdc++ host library.
2. NVCC with VC++, and the VC++ host library.
3. Clang, with the libstdc++ host library.
4. NVRTC, with access to the C host library.

Assuming you compile with `-I<path-to-include/>`:
1. Each header named `<simt/X>` conforms to the specification for the header `<X>` from ISO C++, except that each occurrence of `std::` is prefixed with `simt::`.
2. Except for limitations specified below, each facility thus introduced in `simt::` works in both `__host__` and `__device__` functions, under `-std=c++11` and `-std=c++14`, on Windows and Linux with CUDA 9 or 10 on Volta, Xavier and Turing. (_Though, obviously, not all combinations are possible._)

## Considerations for use under NVRTC

If you choose to use this facility with the [run-time compiler](https://docs.nvidia.com/cuda/nvrtc/index.html) then you will need to ensure it can find the necessary headers. The simplest way to achieve that is to point to headers on your filesystem.

For example, hypothetically speaking, these might be the paths you need to provide:

```c++
  const char *opts[] = {"-std=c++11",
                        "-I/usr/include/linux",
                        "-I/usr/include/c++/7.3.0",
                        "-I/usr/local/cuda/include",
                        "-I/home/olivier/freestanding/include",
                        "--gpu-architecture=compute_70",
                        "--relocatable-device-code=true",
                        "-default-device"};
  nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                                                  8,     // numOptions
                                                  opts); // options
```

## What does not work

In general, for the language support library (`<simt/initializer_list>`, `<simt/new>`, `<simt/typeinfo>`, `<simt/exception>`) the header `<simt/X>` only introduces aliases to the declarations of `<X>` under the `simt::` namespace. Use in `__device__` functions is limited to intrinsic support for the facility by the compiler used.
  
In specific, see the table below.

| Header | Limitation in function | Requires | 
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `<simt/atomic>`           | Except fence functions also add the suffix `_simt`. | `<simt/cstddef>`, `<simt/cstdint>`, `<simt/type_traits>`           |
| `<simt/cfloat>`           |                                                              | `<float.h>`                                                    |
| `<simt/ciso646>`          |                                                              | `<iso646.h>`                                                   |
| `<simt/climits>`          |                                                              | `<limits.h>`                                                   |
| `<simt/cstdalign>`        | Unsupported.                                                | Unsupported.                                                 |
| `<simt/cstdarg>`          | Except `__device__` functions.                                            | `<stdarg.h>`                                                   |
| `<simt/cstdbool>`         |                                                              |                                                              |
| `<simt/cstddef>`          |                                                              | `<stddef.h>`                                                   |
| `<simt/cstdint>`          |                                                              | `<stdint.h>`                                                   |
| `<simt/cstdlib>`          | Except `__device__` functions.                                            | `<stdlib.h>`                                                   |
| `<simt/exception>`        | Except `__device__` functions.                                            | `<simt/cstddef>`, `<simt/cstdint>`, `<simt/type_traits>`           |
| `<simt/initializer_list>` |                                                              | `<simt/cstddef>`                                               |
| `<simt/limits>`           |                                                              | `<simt/type_traits>`                                           |
| `<simt/new>`              | Except `__device__` functions.                                            | `<simt/exception>`, `<simt/type_traits>`, `<simt/cstddef>`, `<simt/cstdlib>` |
| `<simt/type_traits>`      |                                                              | `<simt/cstddef>`                                               |
| `<simt/typeinfo>`         | Except `__device__` functions.                                            | `<simt/exception>`, `<simt/cstddef>`, `<simt/cstdint>`, `<simt/cstdlib>` |
