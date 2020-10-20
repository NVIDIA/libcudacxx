---
parent: API
has_children: true
has_toc: false
nav_order: 0
---

# Synchronization Library

Most synchronization primitives have an additional thread scope template
  parameter, which defines the set of threads that may interact with the object.
Please see the [thread scope section] for more details.

Any header not listed below is omitted.

*: Some the Standard C++ facilities in this header are omitted, see the libcu++
Addendum for details.

{% include_relative synchronization_library/header_table.md %}


[thread scope section]: ./synchronization_library/thread_scopes.md

