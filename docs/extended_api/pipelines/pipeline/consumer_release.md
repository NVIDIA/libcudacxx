---
grand_parent: Pipelines
parent: cuda::pipeline
---

# `cuda::pipeline::consumer_release`

Defined in header `<cuda/pipeline>`:

```cuda
template <cuda::thread_scope Scope>
__host__ __device__
void cuda::pipeline<Scope>::consumer_release();
```

Releases the current _pipeline stage_.

## Expects

The calling thread is a _consumer thread_.

The pipeline is not in a _quitted state_ (see [`cuda::pipeline::quit`]).


[`cuda::pipeline::quit`]: ./quit.md

