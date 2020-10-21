---
nav_exclude: true
---

# cuda::pipeline\<Scope>::**quit**

```c++
bool quit();
```

Quits the current thread's participation in the collective ownership of the corresponding _shared state_ ([`cuda::pipeline_shared_state`](../pipeline_shared_state.md)). Ownership of the _shared state_ is released by the last invoking thread.

## Return value

`true` if ownership of the _shared state_ was released, otherwise `false`.

## Notes

The behavior undefined if any operation other than [`~pipeline`](./destructor.md) is issued by the current thread after quitting.

## Example

```c++
TODO
```
