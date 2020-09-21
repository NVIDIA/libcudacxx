# `<cuda/std/atomic>`, `<cuda/atomic>`

## Extensions in the `cuda::` namespace

The class template `atomic` takes an additional scope argument, defaulted to `thread_scope_system`.

```c++
// This object is atomic for all threads in the system
cuda::atomic<int> a; 

// This object has the same type as the previous
cuda::atomic<int, cuda::thread_scope_system> b; 

// This object is atomic for threads in the same thread block
cuda::atomic<int, cuda::thread_scope_block> c; 
```

The `atomic` class template specializations for integral and pointer types are extended with members `fetch_min` and `fetch_max`. These conform to the requirements in `[atomics.types.int]` and `[atomics.types.pointer]`, with keys _min_ and _max_, and operations `min` and `max`, respectively.

Note that conformance to these requirements include implicit conversions to unsigned types, if applicable.

```c++
cuda::atomic<int> a(1); 
auto x = a.fetch_min(0); // operates as if unsigned
auto y = a.load();
assert(x == 1 && y == 0);
```

## Omissions

None relative to libcxx.

## Restrictions

An object of type `atomic` shall not be accessed concurrently by CPU and GPU threads unless:
  1. it is in managed memory, with `concurrentManagedAccess==1`, or
  2. it is in host memory, with `hostNativeAtomicSupported==1`.

(Note, for objects of scopes other than `thread_scope_system` this is a data-race, and thefore also prohibited regardless of memory characteristics.)

Under Compute Capability 6, an object of type `atomic` may not be used:
  1. with automatic storage duration, or 
  2. if `is_always_lock_free()` returns `false`.

Under Compute Capability prior to 6, objects of type `atomic` may not be used.

## Implementation-Defined Behavior

For each type `T` and scope `S`, the value of `atomic<T,S>::is_always_lock_free()` is as follows:

|Type `T`|Scope `S`|`atomic<T,S>::is_always_lock_free()`|
|-|-|-|
|Any|Any|`sizeof(T) <= 8`|


Objects in namespace `cuda::std::` have the same behavior as corresponding objects in namespace `cuda::` when instantiated with a scope of `cuda::thread_scope_system`.
