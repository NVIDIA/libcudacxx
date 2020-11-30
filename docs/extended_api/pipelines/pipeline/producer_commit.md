---
grand_parent: Pipelines
parent: cuda::pipeline
---

# `cuda::pipeline::producer_commit`

Defined in header `<cuda/pipeline>`:

```cuda
template <cuda::thread_scope Scope>
__host__ __device__
void cuda::pipeline<Scope>::producer_commit();
```

Commits operations previously issued by the current thread to the current
  _pipeline stage_.

## Notes

The calling thread is a _producer thread_.

The pipeline is not in a _quitted state_ (see [`cuda::pipeline::quit`]).


[`cuda::pipeline::quit`]: ./quit.md

