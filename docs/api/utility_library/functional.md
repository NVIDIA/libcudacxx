# `<cuda/std/functional>`

## Omissions

The following facilities ISO/IEC IS 14882 [[functional.syn]] are not available
  in libcu++ today:

- `cuda::std::function`: Polymorphic function object wrapper.
- `cuda::std::bind`: Generic function object binder / lambda facility.
- `cuda::std::reference_wrapper`: Reference wrapper type.
- `cuda::std::hash`: Hash function object.

### `cuda::std::function`

`std::function` is a polymorphic function object wrapper.
Implementing it requires both polymorphism (either hand built dispatch tables
  or the use of C++ virtual functions) and memory allocation.
This means that it is non-trivial to implement a heterogeneous version of this
  facility today.
As such, we have deferred it.

### `cuda::std::bind`

`std::bind` is a general-purpose function object binder / lambda facility.
It relies on constexpr global variables for placeholders, which presents
  heterogeneous implementation challenges today due to how global variables work
  in NVCC.
E.g. We cannot easily ensure the placeholders are the same object with the same
  address in host and device code.
Therefore, we've decided to hold off on providing this feature for now.

### `cuda::std::hash`

`std::hash` is a function object which hashes entities.
While this is an important feature, it is also important that we pick a hash
  implementation that makes sense for GPUs.
That implementation might be
  different from the default that the upstream libc++ uses.
Further research and investigation is required before we can provide this
  feature.

### `cuda::std::reference_wrapper`

`cuda::std::reference_wrapper` is a [*CopyConstructible*] and
  [*CopyAssignable*] wrapper around a reference to an object or function of
  type `T`.
There is nothing that makes this facility difficult to implement heterogeneously
  today.
It is a value type that does not allocate memory, hold
  pointers, have virtual functions, or make calls to platform specific APIs.

No design or functional changes were required to port the upstream `libc++`
  implementations of this facility.
We just had to add execution space specifiers to port it.

However, this feature failed tests involving function pointers with some of the
  compilers we support.
So, we've omitted this feature for now.


[functional.syn]: https://eel.is/c++draft/functional.syn

[*CopyConstructible*]: https://eel.is/c++draft/utility.arg.requirements#:requirements,Cpp17CopyConstructible
[*CopyAssignable*]: https://eel.is/c++draft/utility.arg.requirements#:requirements,Cpp17CopyAssignable
