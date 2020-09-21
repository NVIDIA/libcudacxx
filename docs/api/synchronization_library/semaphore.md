# `<cuda/std/semaphore>`, `<cuda/semaphore>`

## Extensions in the `cuda::` namespace

The class `binary_semaphore` is a class template in this namespace.

The class templates `binary_semaphore` and `counting_semaphore` take an additional scope argument, defaulted to `thread_scope_system`.

```c++
// These objects are semaphores suitable for all threads in the system
cuda::binary_semaphore<> a1; 
cuda::counting_semaphore<> a2; 

// These objects have the same types as the previous (respectively)
cuda::binary_semaphore<cuda::thread_scope_system> b1; 
cuda::counting_semaphore<cuda::thread_scope_system> b2; 

// These objects are semaphores for threads in the same thread block
cuda::binary_semaphore<cuda::thread_scope_block> c1; 
cuda::counting_semaphore<cuda::thread_scope_block> c2; 
```

## Omissions

None.

## Restrictions

An object of type `binary_semaphore` or `counting_semaphore`, shall not be accessed concurrently by CPU and GPU threads unless:
  1. it is in managed memory, with `concurrentManagedAccess==1`, or
  2. it is in host memory, with `hostNativeAtomicSupported==1`.

This requirement is in addition to the requirement to avoid data-races, see [thread scopes]({{ "./thread_scopes.html" | relative_url }}) for more information.

Under Compute Capability 6 or prior, an object of type `binary_semaphore` or `counting_semaphore` may not be used.

## Implementation-Defined Behavior

For each scope `S`, the value of `binary_semaphore<S>::max()` is as follows:

|Scope `S`|`binary_semaphore<S>::max()`|
|-|-|
|Any|`1`|

For each scope `S` and least maximum value `V`, the value of `counting_semaphore<S,V>::max()` is as follows:

|Scope `S`|Least maximum value `V`|`counting_semaphore<S,V>::max()`|
|-|-|-|
|Any|Any|`std::numeric_limits<ptrdiff_t>::max()`|

Objects in namespace `cuda::std::` have the same behavior as corresponding objects in namespace `cuda::` when instantiated with a scope of `cuda::thread_scope_system`.
