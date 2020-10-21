---
nav_exclude: true
---

# cuda::pipeline\<Scope>::**consumer_release**

```c++
void consumer_release();
```

Releases the current _pipeline stage_.

## Notes

Calling this method from a _producer thread_ is undefined behavior.
