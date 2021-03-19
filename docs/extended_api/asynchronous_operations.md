## Asynchronous Operations

Asynchronous operations are performed _as-if_ in other threads. This other threads are related to the thread requesting the asynchronous operations via all scope relationships [^thread_scope].



| [`cuda::memcpy_async`] | Asynchronously copies one range to another. `(function template)` <br/><br/> 1.1.0 / CUDA 11.0 <br/> 1.2.0 / CUDA 11.1 (group & aligned overloads) |


[^thread_scope]: This includes [`cuda::thread_scope_thread`]. That is, [`cuda::thread_scope_thread`] can synchronize multiple threads.

[`cuda::memcpy_async`]: {{ "extended_api/asynchronous_operations/memcpy_async.html" | relative_url }}
[`cuda::thread_scope_thread`]: {{ "extended_api/thread_scopes.html" | relative_url }}
