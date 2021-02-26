---
has_children: true
has_toc: false
nav_order: 2
---

# Standard API

## Standard Library Backports

C++ Standard versions include new language features and new library features. 
As the name implies, language features are new features of the language the require compiler support.
Library features are simply new additions to the Standard Library that typically do not rely on new language features nor require compiler support and could conceivably be implemented in an older C++ Standard. 
Typically, library features are only available in the particular C++ Standard version (or newer) in which they were introduced, even if the library features do not depend on any particular language features.

In effort to make library features available to a broader set of users, the NVIDIA C++ Standard Library relaxes this restriction. 
libcu++ makes a best-effort to provide access to C++ Standard Library features in older C++ Standard versions than they were introduced. 
For example, the calendar functionality added to `<chrono>` in C++20 is made available in C++14. 

Feature availability:
- C++17 an C++20 features of`<chrono>` available in C++14:
  -  calendar functionality, e.g., `year`,`month`,`day`,`year_month_day`
  -  duration functions, e.g., `floor`, `ceil`, `round`
  -  Note: Timezone and clocks added in C++20 are not available
- C++17 features from `<type_traits>` available in C++14:
  - Convenience `_v` aliases such as `is_same_v`
  - `void_t`
  - Trait operations: `conjunction`,`negation`,`disjunction`
  - `invoke_result`



## Synchronization Library

Any Standard C++ header not listed below is omitted.

| [`<cuda/std/atomic>`]    | Atomic objects and operations (see also: [Extended API](./extended_api/synchronization_primitives/atomic.md)). <br/><br/> 1.0.0 / CUDA 10.2 |
| [`<cuda/std/latch>`]     | Single-phase asynchronous thread-coordination mechanism (see also: [Extended API](./extended_api/synchronization_primitives/latch.md)). <br/><br/> 1.1.0 / CUDA 11.0 |
| [`<cuda/std/barrier>`]   | Multi-phase asynchronous thread-coordination mechanism (see also: [Extended API](./extended_api/synchronization_primtives/barrier.md)). <br/><br/> 1.1.0 / CUDA 11.0 |
| [`<cuda/std/semaphore>`] | Primitives for constraining concurrent access (see also: [Extended API](./extended_api/synchronization_primitives/counting_semaphore.md)). <br/><br/> 1.1.0 / CUDA 11.0 |

{% include_relative standard_api/time_library.md %}

{% include_relative standard_api/numerics_library.md %}

{% include_relative standard_api/utility_library.md %}

## C Library

Any Standard C++ header not listed below is omitted.

| [`<cuda/std/cassert>`] | Lightweight assumption testing. <br/><br/> 1.0.0 / CUDA 10.2         |
| [`<cuda/std/cstddef>`] | Fundamental types. <br/><br/> 1.0.0 / CUDA 10.2 <br/> 1.4.0 (`byte`) |


[`<cuda/std/atomic>`]: https://en.cppreference.com/w/cpp/header/atomic
[`<cuda/std/latch>`]: https://en.cppreference.com/w/cpp/header/latch
[`<cuda/std/barrier>`]: https://en.cppreference.com/w/cpp/header/barrier
[`<cuda/std/semaphore>`]: https://en.cppreference.com/w/cpp/header/semaphore
[`<cuda/std/cassert>`]: https://en.cppreference.com/w/cpp/header/cassert
[`<cuda/std/cstddef>`]: https://en.cppreference.com/w/cpp/header/cstddef

