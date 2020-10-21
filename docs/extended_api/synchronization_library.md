---
parent: Extended API
has_children: true
has_toc: false
---

# Synchronization library

The synchronization library provides components for thread and asynchronous operations coordination.

## Synchronization types

| [pipeline](./synchronization_library/pipeline.md)                           | _pipeline_ class template `(class template)`                                       |
| [pipeline_shared_state](./synchronization_library/pipeline_shared_state.md) | _pipeline shared state_ for inter-thread coordination `(class template)`           |
| [pipeline_role](./synchronization_library/pipeline_role.md)                 | defines producer/consumer role for a thread participating in a _pipeline_ `(enum)` |

## Synchronization types factories

| [make_pipeline](./synchronization_library/make_pipeline.md) | creates a _pipeline_ object `(function template)` |

## Operations on synchronization types

| [pipeline_consumer_wait_prior](./synchronization_library/pipeline_consumer_wait_prior.md) | blocks the current thread until all operations committed up to a prior _pipeline stage_ complete `(function template)`|
| [pipeline_producer_commit](./synchronization_library/pipeline_consumer_commit.md)         | Binds operations previously issued by the current thread to a _barrier_ `(function template)`                         |
