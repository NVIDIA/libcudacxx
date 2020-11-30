---
has_children: true
has_toc: false
nav_order: 2
---

# Standard API

Any header not listed below is omitted.

*: Some of the Standard C++ facilities in this header are omitted, see the
libcu++ Addendum for details.

## Synchronization Library

| Header                                         | Added In          | Docs |
|------------------------------------------------|-------------------|------|
| `<cuda/std/atomic>`    | 1.0.0 / CUDA 10.2 | [Standard API](https://en.cppreference.com/w/cpp/header/atomic), [Extended API](./extended_api/atomics.md)       |
| `<cuda/std/barrier>`   | 1.1.0 / CUDA 11.0 | [Standard API](https://en.cppreference.com/w/cpp/header/barrier), [Extended API](./extended_api/barriers.md)     |
| `<cuda/std/latch>`     | 1.1.0 / CUDA 11.0 | [Standard API](https://en.cppreference.com/w/cpp/header/latch), [Extended API](./extended_api/latches.md)        |
| `<cuda/std/semaphore>` | 1.1.0 / CUDA 11.0 | [Standard API](https://en.cppreference.com/w/cpp/header/semaphore), [Extended API](./extended_api/semaphores.md) |

## [Time Library](./standard_api/time_library.md)

{% include_relative standard_api/time_library/header_table.md %}

## [Utility Library](./standard_api/utility_library.md)

{% include_relative standard_api/utility_library/header_table.md %}

## C Library

| Header                   | Added In          | Docs |
|--------------------------|-------------------|------|
| `<cuda/std/cassert>`     | 1.0.0 / CUDA 10.2 | [Standard API](https://en.cppreference.com/w/cpp/header/cassert) |
| `<cuda/std/cfloat>`      | 1.0.0 / CUDA 10.2 | [Standard API](https://en.cppreference.com/w/cpp/header/cfloat)  |
| `<cuda/std/climits>`     | 1.0.0 / CUDA 10.2 | [Standard API](https://en.cppreference.com/w/cpp/header/climits) |
| `<cuda/std/cstddef>`     | 1.0.0 / CUDA 10.2 | [Standard API](https://en.cppreference.com/w/cpp/header/cstddef) |
| `<cuda/std/cstdint>`     | 1.0.0 / CUDA 10.2 | [Standard API](https://en.cppreference.com/w/cpp/header/cstdint) |

