---
grand_parent: Extended API
parent: Asynchronous operations library
---

# cuda::**aligned_size_t**

Defined in header [`<cuda/barrier>`](../headers/barrier.md)

Defined in header [`<cuda/pipeline>`](../headers/pipeline.md)

```c++
template<size_t Alignment>
struct aligned_size_t;
```

The class template `cuda::aligned_size_t` is a _shape_ representing an extent of bytes with a statically defined (address and size) alignment.

## Template parameters

| Alignment | the address & size alignement of the byte extent |

## Data members

| [align](./aligned_size_t/align.md) | the alignment of the byte extent |
| [value](./aligned_size_t/value.md) | the size of the byte extent      |

## Member functions

| [(constructor)](./aligned_size_t/constructor.md) | constructs an _aligned size_                                                      |
| (destructor) [implicitly declared]               | trivial implicit destructor                                                       |
| operator= [implicitly declared]                  | trivial implicit copy/move assignment                                             |
| operator size_t                                  | implicit conversion to [`size_t`](https://en.cppreference.com/w/cpp/types/size_t) |

## Notes

If `value` is not a multiple of `align` the behavior is undefined.

If `Alignment` is not a [valid alignment](https://en.cppreference.com/w/c/language/object#Alignment) the behavior is undefined.

## Example

```c++
#include <cuda/barrier>

__global__ void example_kernel(void * dst, void * src, size_t size)
{
    cuda::barrier<cuda::thread_scope_system> barrier;
    init(&barrier, 1);

    // Implementation cannot make assumptions about alignment
    cuda::memcpy_async(dst, src, size, barrier);

    // Implementation can assume that dst, src and size are 16-bytes aligned and may optimize accordingly
    cuda::memcpy_async(dst, src, cuda::aligned_size_t<16>(size), barrier);

    barrier.arrive_and_wait();
}
```

[See it on Godbolt](https://godbolt.org/z/v7Ev9E){: .btn }
