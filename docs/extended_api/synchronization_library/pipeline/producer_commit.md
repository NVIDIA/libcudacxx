---
nav_exclude: true
---

# cuda::pipeline\<Scope>::**producer_commit**

```c++
void producer_commit();
```

Commits operations previously issued by the current thread to the current _pipeline stage_.

## Notes

Calling this method from a _consumer thread_ is undefined behavior.
