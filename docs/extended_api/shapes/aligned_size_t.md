---
grand_parent: Extended API
parent: Shapes
---

# `cuda::aligned_size_t`

Defined in headers `<cuda/barrier>` and `<cuda/pipeline>`:

```cuda
template <cuda::std::size_t Alignment>
struct cuda::aligned_size_t {
  static constexpr cuda::std::size_t align = Align;
  cuda::std::size_t value;
  __host__ __device__ explicit constexpr aligned_size(cuda::std::size_t size);
  __host__ __device__ constexpr operator cuda::std::size_t();
};
```

The class template `cuda::aligned_size_t` is a _shape_ representing an extent
  of bytes with a statically defined (address and size) alignment.

## Template Parameters

| `Alignment` | The address and size alignement of the byte extent. |

## Data Members

| `align` | The alignment of the byte extent. |
| `value` | The size of the byte extent.      |

## Member Functions

| (constructor)                      | Constructs an _aligned size_. If the `size` is not a multiple of `Alignment` the behavior is undefined. |
| (destructor) [implicitly declared] | Trivial implicit destructor.                                                                            |
| `operator=` [implicitly declared]  | Trivial implicit copy/move assignment.                                                                  |
| `operator cuda::std::size_t`       | Implicit conversion to [`cuda::std::size_t`].                                                           |

## Notes

If `Alignment` is not a [valid alignment], the behavior is undefined.

## Example

```cuda
#include <cuda/barrier>

__global__ void example_kernel(void* dst, void* src, size_t size) {
  cuda::barrier<cuda::thread_scope_system> bar;
  init(&bar, 1);

  // Implementation cannot make assumptions about alignment.
  cuda::memcpy_async(dst, src, size, bar);

  // Implementation can assume that dst, src and size are 16-bytes aligned and
  // may optimize accordingly.
  cuda::memcpy_async(dst, src, cuda::aligned_size_t<16>(size), bar);

  bar.arrive_and_wait();
}
```

[See it on Godbolt](https://godbolt.org/z/jr8GqT){: .btn }


[valid alignment]: https://en.cppreference.com/w/c/language/object#Alignment

[`cuda::std::size_t`]: https://en.cppreference.com/w/cpp/types/size_t

