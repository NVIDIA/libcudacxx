---
nav_exclude: true
---

# cuda::aligned_size_t\<Alignment>::**aligned_size_t**

```c++
explicit aligned_size_t(size_t size);
```

Constructs an `aligned_size_t` _shape_.

## Notes

If `size` is not a multiple of `Alignment` the behavior is undefined.
