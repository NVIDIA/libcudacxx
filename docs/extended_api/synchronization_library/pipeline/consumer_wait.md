---
nav_exclude: true
---

# cuda::pipeline\<Scope>::**consumer_wait**, cuda::pipeline\<Scope>::**consumer_wait_for**, cuda::pipeline\<Scope>::**consumer_wait_until**

```c++
void consumer_wait();                                                                  // (1)

template<class Rep, class Period>
bool consumer_wait_for(const std::chrono::duration<Rep, Period> & duration);           // (2)

template<class Clock, class Duration>
bool consumer_wait_until(const std::chrono::time_point<Clock, Duration> & time_point); // (3)
```

1. blocks the current thread until all operations committed to the current _pipeline stage_ complete
2. blocks the current thread until all operations committed to the current _pipeline stage_ complete or after the specified timeout duration
3. blocks the current thread until all operations committed to the current _pipeline stage_ complete or until specified time point has been reached

## Parameters

| duration   | an object of type `cuda::std::chrono::duration` representing the maximum time to spend waiting |
| time_point | an object of type `cuda::std::chrono::time_point` representing the time when to stop waiting   |

## Return value

`false` if the _wait_ timed out, `true` otherwise.

## Notes

If this method is called from a _producer thread_ the behavior is undefined.

If the pipeline is in a _quitted state_ (see [`pipeline::quit`](./quit.md)), the behavior is undefined.
