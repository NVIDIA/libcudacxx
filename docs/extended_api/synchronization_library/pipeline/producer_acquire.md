---
nav_exclude: true
---

# cuda::pipeline\<Scope>::**producer_acquire**

```c++
void producer_acquire();
```

Blocks the current thread until the next _pipeline stage_ is available.

## Notes

If this method is called from a _consumer thread_ the behavior is undefined.

If the pipeline is in a _quitted state_ (see [`pipeline::quit`](./quit.md)), the behavior is undefined.
