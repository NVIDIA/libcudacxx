---
grand_parent: Extended API
parent: Concepts
---

# Group

```c++
struct Group {
  static constexpr cuda::thread_scope thread_scope;
  integral size() const;
  integral thread_rank() const;
  void sync() const;
};
```

The _Group concept_ defines the requirements of a type that represents a group of cooperating threads.

## Data members

| thread_scope | the scope at which `Group::sync()` synchronizes memory operations and thread execution |

## Member functions

| size        | returns the number of participating threads                                                        |
| thread_rank | returns a unique value for each participating thread (`0 <= Group::thread_rank() < Group::size()`) |
| sync        | synchronizes the participating threads                                                             |

## Notes

This concept is defined for documentation purposes but is not materialized in the library.

## Example

```c++
#include <cuda/atomic>

struct single_thread_group {
    static constexpr cuda::thread_scope thread_scope = cuda::thread_scope::thread_scope_thread;
    size_t size() const { return 1; }
    size_t thread_rank() const { return 0; }
    void sync() const { }
};
```

[See it on Godbolt](https://godbolt.org/z/453r3s){: .btn }
