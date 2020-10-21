---
grand_parent: Extended API
parent: Concepts
---

# Shape

```c++
struct Shape {
    operator size_t() const;
};
```

The _Shape concept_ defines the requirements of a type that represents a byte extent with a particular memory layout.

## Member functions

| operator size_t | implicit conversion to [`size_t`](https://en.cppreference.com/w/cpp/types/size_t) |

## Notes

This concept is defined for documentation purposes but is not materialized in the library.

## Example

```c++
// A size that carries an alignment hint
template <size_t Align>
struct aligned_size {
    static constexpr size_t align = Align;
    size_t size;
    aligned_size(size_t s) : size(s) {}
    operator size_t() const { return size; }
};
```

[See it on Godbolt](https://godbolt.org/z/hbajKo){: .btn }
