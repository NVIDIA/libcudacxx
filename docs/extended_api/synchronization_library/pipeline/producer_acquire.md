---
nav_exclude: true
---

# cuda::pipeline\<Scope>::**producer_acquire**

```c++
void producer_acquire();
```

Blocks the current thread until the next _pipeline stage_ is available.

## Notes

Calling this method from a _consumer thread_ is undefined behavior.
