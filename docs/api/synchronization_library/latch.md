# `<cuda/std/latch>`, `<cuda/latch>`

## Extensions in the `cuda::` namespace

The class template `latch` takes an additional scope argument, defaulted to `thread_scope_system`.

```c++
// This object is latch suitable for all threads in the system
cuda::latch<> a; 

// This object has the same type as the previous
cuda::latch<cuda::thread_scope_system> b; 

// This object is a latch for threads in the same thread block
cuda::latch<cuda::thread_scope_block> c; 
```

## Omissions

None.

## Restrictions

An object of type `latch` shall not be accessed concurrently by CPU and GPU threads unless:
  1. it is in managed memory, with `concurrentManagedAccess==1`, or
  2. it is in host memory, with `hostNativeAtomicSupported==1`.

This requirement is in addition to the requirement to avoid data-races, see [thread scopes]({{ "./thread_scopes.html" | relative_url }}) for more information.

Under Compute Capability 6 or prior, an object of type `latch` may not be used.

## Implementation-Defined Behavior

For each scope `S`, the value of `latch<S>::max()` is as follows:

|Scope `S`|`latch<S>::max()`|
|-|-|
|Any|`std::numeric_limits<ptrdiff_t>::max()`|

Objects in namespace `cuda::std::` have the same behavior as corresponding objects in namespace `cuda::` when instantiated with a scope of `cuda::thread_scope_system`.
