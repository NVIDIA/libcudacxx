---
grand_parent: Extended API
parent: Asynchronous operations library
---

# cuda::**memcpy_async**

Defined in header [`<cuda/barrier>`](../../api/synchronization_library/barrier.md)

```c++
template<typename Shape, thread_scope Scope, typename CompletionFunction>
void memcpy_async(void * destination, void const * source, Shape size, barrier<Scope, CompletionFunction> & barrier);                      // (1)

template<typename Group, typename Shape, thread_scope Scope, typename CompletionFunction>
void memcpy_async(Group const & group, void * destination, void const * source, Shape size, barrier<Scope, CompletionFunction> & barrier); // (2)
```

Defined in header [`<cuda/pipeline>`](../headers/pipeline.md)

```c++
template<typename Shape, thread_scope Scope>
void memcpy_async(void * destination, void const * source, Shape size, pipeline<Scope> & pipeline);                                        // (3)

template<typename Group, typename Shape, thread_scope Scope>
void memcpy_async(Group const & group, void * destination, void const * source, Shape size, pipeline<Scope> & pipeline);                   // (4)
```

Asynchronously copies `size` bytes from the memory location pointed to by `source` to the memory location pointed to by `destination`.
Both objects are reinterpreted as arrays of `unsigned char`.

`cuda::memcpy_async` have similar constraints to [`std::memcpy`](https://en.cppreference.com/w/cpp/string/byte/memcpy), namely:
* If the objects overlap, the behavior is undefined.
* If either `destination` or `source` is an invalid or null pointer, the behavior is undefined (even if `count` is zero).
* If the objects are [potentially-overlapping](https://en.cppreference.com/w/cpp/language/object#Subobjects) the behavior is undefined.
* If the objects are not of [`TriviallyCopyable`](https://en.cppreference.com/w/cpp/named_req/TriviallyCopyable) type the program is ill-formed, no diagnostic required.

If _Shape_ is [`cuda::aligned_size_t`](./aligned_size_t.md)), `source` and `destination` are both required to be aligned on [`cuda::aligned_size_t::align`](./aligned_size_t/align.md), else the behavior is undefined.

If `pipeline` is in a _quitted state_ (see [`pipeline::quit`](../synchronization_library/pipeline/quit.md)), the behavior is undefined.

1. Binds the asynchronous copy completion to `barrier` and issues the copy in the current thread.
2. Binds the asynchronous copy completion to `barrier` and cooperatively issues the copy across all threads in `group`.
3. Binds the asynchronous copy completion to `pipeline` and issues the copy in the current thread
4. Binds the asynchronous copy completion to `pipeline` and cooperatively issues the copy across all threads in `group`.

## Template parameters

| Group | a type satisfying the [_group concept_](../concepts/group.md)                                                                                                                  |
| Shape | a type satisfying the [_shape concept_](../concepts/shape.md) (see [`size_t`](https://en.cppreference.com/w/c/types/size_t) and [`cuda::aligned_size_t`](./aligned_size_t.md)) |

## Parameters

| group       | the group of threads                                    |
| destination | pointer to the memory location to copy to               |
| source      | pointer to the memory location to copy from             |
| size        | the number of bytes to copy                             |
| barrier     | the barrier object used to wait on the copy completion  |
| pipeline    | the pipeline object used to wait on the copy completion |

## Example

```c++
TODO
```
