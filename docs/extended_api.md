---
has_children: true
has_toc: false
nav_order: 3
---

:warning: **The libcudacxx repository has been archived and is now part of the unified [nvidia/cccl repository](https://github.com/nvidia/cccl). See the [announcement here](https://github.com/NVIDIA/cccl/discussions/520) for more information. Please visit the new repository for the latest updates. The new documentation can be found [here](https://nvidia.github.io/cccl/libcudacxx/).** :warning:

# Extended API

## Fundamentals

| [Thread Scopes]               | Defines the kind of threads that can synchronize using a primitive. `(enum)` <br/><br/> 1.0.0 / CUDA 10.2 |
| [Thread Groups]               | Concepts for groups of cooperating threads. `(concept)`                      <br/><br/> 1.2.0 / CUDA 11.1 |

{% include_relative extended_api/shapes.md %}

{% include_relative extended_api/synchronization_primitives.md %}

{% include_relative extended_api/asynchronous_operations.md %}

{% include_relative extended_api/memory_access_properties.md %}

{% include_relative extended_api/functional.md %}

[Thread Scopes]: ./extended_api/memory_model.md#thread-scopes
[Thread Groups]: ./extended_api/thread_groups.md

