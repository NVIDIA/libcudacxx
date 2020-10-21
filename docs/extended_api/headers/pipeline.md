---
grand_parent: Extended API
parent: Headers
---

# \<cuda/**pipeline**>

This header is part of the [synchronization library](../synchronization_library.md).

## Classes

| [aligned_size_t](../asynchronous_operations_library/aligned_size_t.md)       | defines an extent of bytes with a statically defined alignment `(class template)`  |
| [pipeline](../synchronization_library/pipeline.md)                           | _pipeline_ class template `(class template)`                                       |
| [pipeline_shared_state](../synchronization_library/pipeline_shared_state.md) | _pipeline shared state_ for inter-thread coordination `(class template)`           |
| [pipeline_role](../synchronization_library/pipeline_role.md)                 | defines producer/consumer role for a thread participating in a _pipeline_ `(enum)` |

## Functions

| [make_pipeline](../synchronization_library/make_pipeline.md)                               | creates a _pipeline_ object `(function template)`                                                                     |
| [pipeline_consumer_wait_prior](../synchronization_library/pipeline_consumer_wait_prior.md) | blocks the current thread until all operations committed up to a prior _pipeline stage_ complete `(function template)`|
| [pipeline_producer_commit](../synchronization_library/pipeline_producer_commit.md)         | binds operations previously issued by the current thread to a _barrier_ `(function template)`                         |
| [memcpy_async](../asynchronous_operations_library/memcpy_async.md)                         | asynchronously copies one buffer to another `(function template)`                                                     |

## Synopsis

```c++
namespace cuda {
    template<size_t Alignment>
    struct aligned_size_t;

    enum class pipeline_role : /* unspecified */ {
        producer,
        consumer
    };

    template<thread_scope Scope, uint8_t StagesCount>
    class pipeline_shared_state;

    template<thread_scope Scope>
    class pipeline;

    pipeline<thread_scope_thread> make_pipeline();

    template<class Group, thread_scope Scope, uint8_t StagesCount>
    pipeline<Scope> make_pipeline(const Group & group, pipeline_shared_state<Scope, StagesCount> * shared_state);

    template<class Group, thread_scope Scope, uint8_t StagesCount>
    pipeline<Scope> make_pipeline(const Group & group, pipeline_shared_state<Scope, StagesCount> * shared_state, size_t producer_count);

    template<class Group, thread_scope Scope, uint8_t StagesCount>
    pipeline<Scope> make_pipeline(const Group & group, pipeline_shared_state<Scope, StagesCount> * shared_state, pipeline_role role);

    template<uint8_t Prior>
    void pipeline_consumer_wait_prior(pipeline<thread_scope_thread> & pipeline);

    template<thread_scope Scope>
    void pipeline_producer_commit(pipeline<thread_scope_thread> & pipeline, barrier<Scope> & barrier);

    template<typename Shape, thread_scope Scope>
    void memcpy_async(void * destination, void const * source, Shape size, pipeline<Scope> & pipeline);

    template<typename Group, typename Shape, thread_scope Scope>
    void memcpy_async(Group const & group, void * destination, void const * source, Shape size, pipeline<Scope> & pipeline);
}
```

## Class template `cuda::aligned_size_t`

```c++
template<size_t Alignment>
struct aligned_size_t {
    static constexpr size_t align = Alignment;
    size_t value;
    explicit aligned_size_t(size_t size);
    operator size_t() const;
};
```


## Class template `cuda::pipeline_shared_state`

```c++
namespace cuda {
    template<thread_scope Scope, uint8_t StagesCount>
    class pipeline_shared_state {
        pipeline_shared_state() = default;
        pipeline_shared_state(const pipeline_shared_state &) = delete;
        pipeline_shared_state(pipeline_shared_state &&) = delete;
        pipeline_shared_state & operator=(pipeline_shared_state &&) = delete;
        pipeline_shared_state & operator=(const pipeline_shared_state &) =  delete;
    };
}
```

## Class template `cuda::pipeline`

```c++
namespace cuda {
    template<thread_scope Scope>
    class pipeline {
        pipeline(pipeline &&) = default;
        pipeline(const pipeline &) = delete;
        pipeline & operator=(pipeline &&) = delete;
        pipeline & operator=(const pipeline &) = delete;
        ~pipeline();

        void producer_acquire();
        void producer_commit();
        void consumer_wait();
        template<class Rep, class Period>
        bool consumer_wait_for(const std::chrono::duration<Rep, Period> & duration);
        template<class Clock, class Duration>
        bool consumer_wait_until(const std::chrono::time_point<Clock, Duration> & time_point);
        void consumer_release();

        bool quit();
    };
}
```
