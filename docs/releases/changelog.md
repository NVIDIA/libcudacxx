---
parent: Releases
nav_order: 0
---

# Changelog

## libcu++ 1.3.0 (CUDA Toolkit 11.2)

libcu++ 1.3.0 is a major release included in the CUDA 11.2 toolkit. It supports
ABI version 2 and 3. It adds `<tuple>` and `pair`, although they are not
supported with NVCC + MSVC. It also adds documentation.

### New Features

- `<cuda/std/tuple>`/`cuda::std::tuple`.
  - Not supported with NVCC + MSVC.
- `<cuda/std/utility>`/`cuda::std::pair`.
  - Not supported with NVCC + MSVC.

### Other Enhancements

- [Documentation](https://nvidia.github.io/libcudacxx).

### Bug Fixes

- #21: Disable `__builtin_is_constant_evaluated` usage with NVCC in C++11 mode
    because it's broken.
- #25: Fix some declarations/definitions in `__threading_support` which have
    inconsistent qualifiers.
  Thanks to Gonzalo Brito Gadeschi for this contribution.

## libcu++ 1.2.0 (CUDA Toolkit 11.1)

TODO

## libcu++ 1.1.1 (CUDA Toolkit 11.0)

TODO

## libcu++ 1.1.0 (CUDA Toolkit 11.0 Early Access)

TODO

## libcu++ 1.0.0 (CUDA Toolkit 10.2)

TODO

TODO: Warn about Debian package bug.

