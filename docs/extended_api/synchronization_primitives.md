---
parent: Extended API
has_children: true
has_toc: false
nav_order: 3
---

# Synchronization Primitives

## Atomics

| [`cuda::atomic`]             | System-wide [`cuda::std::atomic`] objects. `(class template)` |

## Barriers

| [`cuda::barrier`]            | System-wide [`cuda::std::barrier`] objects. `(class template)` |

## Latches

| [`cuda::latch`]              | System-wide [`cuda::std::latch`] objects. `(class template)` |

## Semaphores

| [`cuda::counting_semaphore`] | System-wide [`cuda::std::counting_semaphore`] objects. `(class template)` |
| [`cuda::binary_semaphore`]   | System-wide [`cuda::std::binary_semaphore`] objects. `(class template)` |

## Pipelines

The pipeline library is included in the CUDA Toolkit, but is not part of the
  open source libcu++ distribution.

### Pipeline Types

| [`cuda::pipeline`]              | Coordination mechanism for sequencing asynchronous operations. `(class template)`   |
| [`cuda::pipeline_shared_state`] | [`cuda::pipeline`] shared state object. `(class template)`                          |
| [`cuda::pipeline_role`]         | Defines producer/consumer role for a thread participating in a _pipeline_. `(enum)` |

### Pipeline Factories

| [`cuda::make_pipeline`] | Creates a [`cuda::pipeline`]. `(function template)` |

### Pipeline Operations

| [`cuda::pipeline_consumer_wait_prior`] | Blocks the current thread until all operations committed up to a prior _pipeline stage_ complete. `(function template)` |
| [`cuda::pipeline_producer_commit`]     | Binds operations previously issued by the current thread to a [`cuda::barrier`]. `(function template)`                  |


[`cuda::std::atomic`]: https://en.cppreference.com/w/cpp/atomic/atomic
[`cuda::std::barrier`]: https://en.cppreference.com/w/cpp/thread/barrier
[`cuda::std::latch`]: https://en.cppreference.com/w/cpp/thread/latch
[`cuda::std::counting_semaphore`]: https://en.cppreference.com/w/cpp/thread/counting_semaphore
[`cuda::std::binary_semaphore`]: https://en.cppreference.com/w/cpp/thread/binary_semaphore

[asynchronous operations]: ./asynchronous_operations.md
[`cuda::memcpy_async`]: ./asynchronous_operations/memcpy_async.md

[`cuda::atomic`]: ./synchronization_primitives/atomic.md
[`cuda::barrier`]: ./synchronization_primitives/barrier.md
[`cuda::latch`]: ./synchronization_primitives/latch.md
[`cuda::counting_semaphore`]: ./synchronization_primitives/counting_semaphore.md
[`cuda::binary_semaphore`]: ./synchronization_primitives/binary_semaphore.md

[`cuda::pipeline`]: ./synchronization_primitives/pipeline.md
[`cuda::pipeline_shared_state`]: ./synchronization_primitives/pipeline_shared_state.md
[`cuda::pipeline_role`]: ./synchronization_primitives/pipeline_role.md
[`cuda::make_pipeline`]: ./synchronization_primitives/make_pipeline.md
[`cuda::pipeline_consumer_wait_prior`]: ./synchronization_primitives/pipeline_consumer_wait_prior.md
[`cuda::pipeline_producer_commit`]: ./synchronization_primitives/pipeline_producer_commit.md

