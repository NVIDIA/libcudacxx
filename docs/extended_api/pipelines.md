---
parent: Extended API
has_children: true
has_toc: false
nav_order: 7
---

# Pipelines

Pipelines are coordination mechanisms that sequence [asynchronous operations],
  such as [`cuda::memcpy_async`], into multiple stages.

The pipeline library is included in the CUDA Toolkit, but is not part of the
  open source libcu++ distribution.

## Pipeline Types

| [`cuda::pipeline`]              | Coordination mechanism for sequencing asynchronous operations. `(class template)`   |
| [`cuda::pipeline_shared_state`] | [`cuda::pipeline`] shared state object. `(class template)`                          |
| [`cuda::pipeline_role`]         | Defines producer/consumer role for a thread participating in a _pipeline_. `(enum)` |

## Pipeline Factories

| [`cuda::make_pipeline`] | Creates a [`cuda::pipeline`]. `(function template)` |

## Pipeline Operations

| [`cuda::pipeline_consumer_wait_prior`] | Blocks the current thread until all operations committed up to a prior _pipeline stage_ complete. `(function template)` |
| [`cuda::pipeline_producer_commit`]     | Binds operations previously issued by the current thread to a [`cuda::barrier`]. `(function template)`                  |


[asynchronous operations]: ./asynchronous_operations.md
[`cuda::memcpy_async`]: ./asynchronous_operations/memcpy_async.md

[`cuda::barrier`]: ./barriers.md

[`cuda::pipeline`]: ./pipelines/pipeline.md
[`cuda::pipeline_shared_state`]: ./pipelines/pipeline_shared_state.md
[`cuda::pipeline_role`]: ./pipelines/pipeline_role.md
[`cuda::make_pipeline`]: ./pipelines/make_pipeline.md
[`cuda::pipeline_consumer_wait_prior`]: ./pipelines/pipeline_consumer_wait_prior.md
[`cuda::pipeline_producer_commit`]: ./pipelines/pipeline_producer_commit.md

