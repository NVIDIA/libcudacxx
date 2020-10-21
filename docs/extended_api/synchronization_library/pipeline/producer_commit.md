---
nav_exclude: true
---

# cuda::pipeline\<Scope>::**producer_commit**

```c++
void producer_commit();
```

Commits operations previously issued by the current thread to the current _pipeline stage_.

## Notes

If this method is called from a _consumer thread_ the behavior is undefined.

If the pipeline is in a _quitted state_ (see [`pipeline::quit`](./quit.md)), the behavior is undefined.
