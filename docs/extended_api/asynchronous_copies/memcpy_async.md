---
grand_parent: Extended API
parent: Asynchronous Operations
---

# `cuda::memcpy_async`

Defined in header `<cuda/barrier>`:

```cuda
// (1)
template <typename Shape, cuda::thread_scope Scope, typename CompletionFunction>
__host__ __device__
void cuda::memcpy_async(void* destination, void const* source, Shape size,
                        cuda::barrier<Scope, CompletionFunction>& barrier);

// (2)
template <typename Group,
          typename Shape, cuda::thread_scope Scope, typename CompletionFunction>
__host__ __device__
void cuda::memcpy_async(Group const& group,
                        void* destination, void const* source, Shape size,
                        cuda::barrier<Scope, CompletionFunction>& barrier);
```

Defined in header `<cuda/pipeline>`:

```cuda
// (3)
template <typename Shape, cuda::thread_scope Scope>
void cuda::memcpy_async(void* destination, void const* source, Shape size,
                        cuda::pipeline<Scope>& pipeline);

// (4)
template <typename Group, typename Shape, cuda::thread_scope Scope>
void cuda::memcpy_async(Group const& group,
                        void * destination, void const* source, Shape size,
                        cuda::pipeline<Scope>& pipeline);
```

`cuda::memcpy_async` asynchronously copies `size` bytes from the memory
  location pointed to by `source` to the memory location pointed to by
  `destination`.
Both objects are reinterpreted as arrays of `unsigned char`.

1. Binds the asynchronous copy completion to `cuda::barrier` and issues the copy
   in the current thread.
2. Binds the asynchronous copy completion to `cuda::barrier` and cooperatively
   issues the copy across all threads in `group`.
3. Binds the asynchronous copy completion to `cuda::pipeline` and issues the copy
   in the current thread
4. Binds the asynchronous copy completion to `cuda::pipeline` and cooperatively
   issues the copy across all threads in `group`.

## Notes

`cuda::memcpy_async` have similar constraints to [`std::memcpy`], namely:
* If the objects overlap, the behavior is undefined.
* If either `destination` or `source` is an invalid or null pointer, the
    behavior is undefined (even if `count` is zero).
* If the objects are [potentially-overlapping] the behavior is undefined.
* If the objects are not of [_TriviallyCopyable_] type the program is
    ill-formed, no diagnostic required.

If _Shape_ is [`cuda::aligned_size_t`], `source` and `destination` are both
  required to be aligned on `cuda::aligned_size_t::align`, else the behavior is
  undefined.

If `cuda::pipeline` is in a _quitted state_ (see [`cuda::pipeline::quit`]), the
  behavior is undefined.

For cooperative variants, if the parameters are not the same across all threads
  in `group`, the behavior is undefined.

## Template Parameters

| `Group` | A type satisfying the [_Group_] concept.                  |
| `Shape` | Either [`cuda::std::size_t`] or [`cuda::aligned_size_t`]. |

## Parameters

| `group`       | The group of threads.                                    |
| `destination` | Pointer to the memory location to copy to.               |
| `source`      | Pointer to the memory location to copy from.             |
| `size`        | The number of bytes to copy.                             |
| `barrier`     | The barrier object used to wait on the copy completion.  |
| `pipeline`    | The pipeline object used to wait on the copy completion. |

## Examples

```cuda
#include <cuda/barrier>

__global__ void example_kernel(char* dst, char* src) {
  cuda::barrier<cuda::thread_scope_system> bar;
  init(&bar, 1);

  cuda::memcpy_async(dst,     src,      1, bar);
  cuda::memcpy_async(dst + 1, src + 8,  1, bar);
  cuda::memcpy_async(dst + 2, src + 16, 1, bar);
  cuda::memcpy_async(dst + 3, src + 24, 1, bar);

  bar.arrive_and_wait();
}
```

[See it on Godbolt](https://godbolt.org/z/7nqq9j){: .btn }


[`std::memcpy`]: https://en.cppreference.com/w/cpp/string/byte/memcpy

[potentially-overlapping]: https://en.cppreference.com/w/cpp/language/object#Subobjects

[_TriviallyCopyable_]: https://en.cppreference.com/w/cpp/named_req/TriviallyCopyable

[_ThreadGroup_]: ./thread_group.md

[`cuda::std::size_t`]: https://en.cppreference.com/w/c/types/size_t
[`cuda::aligned_size_t`]: ./shapes/aligned_size_t.md

[`cuda::pipeline::quit`]: ./pipelines/pipeline/quit.md
