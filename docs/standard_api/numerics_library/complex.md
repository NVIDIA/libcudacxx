---
grand_parent: Standard API
parent: Numerics Library
nav_order: 2
---

# `<cuda/std/complex>`

## Omissions

When using libcu++ with NVCC, `complex` does not support `long double` or
  `complex` literals (`_i`, `_if`, and `_il`).
NVCC warns on any usage of `long double` in device code, because `long double`
  will be demoted to `double` in device code.
This warning can be suppressed silenced with `#pragma`s, but only globally, not
  just when using `complex`.
User-defined floating-point literals must be specified in terms of
  `long double`, so they lead to warns that are unable to be suppressed.

## Customizations

Our implementation by default normalizes denormals during multiplication and division. This adds a significant runtime overhead, so we allow the user to disable that normalization if it is not desired.

A user can define `LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS` to disable normalization for both multiplication *and* division.

A user can also define `LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_DIVISION` or `LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_DIVISION` to disable normalization for multiplication or division individualy.

