---
nav_exclude: true
---

# cuda::pipeline\<Scope>::**consumer_release**

```c++
void consumer_release();
```

Releases the current _pipeline stage_.

## Notes

If this method is called from a _producer thread_ the behavior is undefined.

If the pipeline is in a _quitted state_ (see [`pipeline::quit`](./quit.md)), the behavior is undefined.
