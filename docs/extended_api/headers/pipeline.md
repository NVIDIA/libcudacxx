---
grand_parent: Extended API
parent: Headers
---

# \<cuda/**pipeline**>

This header is part of the [synchronization library](../synchronization_library.md).

## Classes

| [pipeline](../synchronization_library/pipeline/pipeline.md)                           | _pipeline_ class template `(class template)`                                       |
| [pipeline_shared_state](../synchronization_library/pipeline/pipeline_shared_state.md) | _pipeline shared state_ for inter-thread coordination `(class template)`           |
| [pipeline_role](../synchronization_library/pipeline/pipeline_role.md)                 | defines producer/consumer role for a thread participating in a _pipeline_ `(enum)` |

## Functions

| [make_pipeline](../synchronization_library/pipeline/make_pipeline.md)                               | creates a _pipeline_ object                                                                      |
| [pipeline_consumer_wait_prior](../synchronization_library/pipeline/pipeline_consumer_wait_prior.md) | blocks the current thread until all operations committed up to a prior _pipeline stage_ complete |
| [pipeline_producer_commit](./synchronization_library/pipeline_consumer_commit.md)                   | Binds operations previously issued by the current thread to a _barrier_                          |

## Synopsis

```c++
namespace cuda {
    enum pipeline_role : /* unspecified */ {
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

    template<typename Group, typename Size, thread_scope Scope>
    void memcpy_async(Group const & group, void * destination, void const * source, Size size, pipeline<Scope> & pipeline);

    template<typename Size, thread_scope Scope>
    void memcpy_async(void * destination, void const * source, Size size, pipeline<Scope> & pipeline);
}
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
