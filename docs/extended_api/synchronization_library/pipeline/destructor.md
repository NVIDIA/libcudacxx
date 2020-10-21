---
nav_exclude: true
---

# cuda::pipeline\<Scope>::**~pipeline**

```c++
~pipeline();
```

Destructs the pipeline.

## Notes

Calls [`cuda::pipeline<scope>::quit`](./quit.md) if it was not called by the current thread.
