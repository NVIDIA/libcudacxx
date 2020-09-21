# `<cuda/std/barrier>`, `<cuda/barrier>`

## Extensions in the `cuda::` namespace

The class template `barrier` takes an additional scope argument, defaulted to `thread_scope_system`.

```c++
// This object is barrier suitable for all threads in the system
cuda::barrier<> a; 

// This object has the same type as the previous
cuda::barrier<cuda::thread_scope_system> b; 

// This object is a barrier for threads in the same thread block
cuda::barrier<cuda::thread_scope_block> c; 
```

The class template `barrier` may also be declared without initialization; a new static member `init` that be used to initialize the object.

```c++
__shared__ cuda::barrier<cuda::thread_scope_block> b; // shared memory does not allow initialization

init(&b, 1); // use this free function to initialize this object
```

In the `device::` sub-namespace a `__device__` free function is available that provides direct access to the underlying PTX state of a `barrier` object, if its scope is `thread_scope_block` and it is allocated in shared memory.

```c++
namespace cuda { namespace device {

__device__ std::uint64_t* barrier_native_handle(
    barrier<thread_scope_block>& b);

}}
```
* Expects: `b` is in shared memory;
* Returns: a pointer to the PTX "mbarrier" subobject of the `barrier` object.

For example:

```c++
auto ptr = barrier_native_handle(b);

asm volatile (
    "mbarrier.arrive.b64 _, [%0];"
    :: "l"(ptr)
    : "memory"); 
// equivalent to: (void)b.arrive();
```

## Omissions

None.

## Restrictions

An object of type `barrier` shall not be accessed concurrently by CPU and GPU threads unless:
  1. it is in managed memory, with `concurrentManagedAccess==1`, or
  2. it is in host memory, with `hostNativeAtomicSupported==1`.

(Note, for objects of scopes other than `thread_scope_system` this is a data-race, and thefore also prohibited regardless of memory characteristics.)

Under Compute Capability 6 or prior, an object of type `barrier` may not be used.

## Implementation-Defined Behavior

For each scope `S` and completion function `F`, the value of `barrier<S,F>::max()` is as follows:

|Scope `S`|Completion function `F`|`barrier<S,F>::max()`|
|-|-|-|
|`thread_scope_block`|Default or user-provided|`(1 << 20) - 1`|
|Not `thread_scope_block`|Default|`std::numeric_limits<int32_t>::max()`|
|Not `thread_scope_block`|User-provided|`std::numeric_limits<ptrdiff_t>::max()`|

Objects in namespace `cuda::std::` have the same behavior as corresponding objects in namespace `cuda::` when instantiated with a scope of `cuda::thread_scope_system`.
