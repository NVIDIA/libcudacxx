//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

#include <type_traits>
#include <cuda/atomic>
#include <cuda_runtime_api.h>
#include <iostream>

using std::uint8_t;
using std::uint16_t;
using std::uint32_t;
using std::uint64_t;
using std::int8_t;
using std::int16_t;
using std::int32_t;
using std::int64_t;

#define nthreads 64

#define HETEROGENEOUS_SAFE_CALL(...)					\
  do {									\
      cudaError_t err = __VA_ARGS__;					\
	if (err != cudaSuccess)						\
	  {								\
	    fprintf(stderr, "CUDA ERROR[%d]: %s: %s\n",			\
		    __LINE__, cudaGetErrorName(err), cudaGetErrorString(err)); \
	    std::abort();						\
	  }								\
  } while (false)


template <typename In, typename Out, typename Test>
void test(In&& in, Out&& out, Test uop) {
  using T = std::decay_t<decltype(in(0))>;

  T* data_d;
  HETEROGENEOUS_SAFE_CALL(cudaMalloc((void**)&data_d, sizeof(T) * nthreads));

  T* data_h = new T[nthreads];
  for (int i = 0; i < nthreads; ++i) data_h[i] = in(i);

  HETEROGENEOUS_SAFE_CALL(cudaMemcpy(data_d, data_h, sizeof(T) * nthreads, cudaMemcpyHostToDevice));

  uop<<<1, nthreads>>>(data_d);
  HETEROGENEOUS_SAFE_CALL(cudaGetLastError());
  HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());
  HETEROGENEOUS_SAFE_CALL(cudaMemcpy(data_h, data_d, sizeof(T) * nthreads, cudaMemcpyDeviceToHost));
  HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());

  bool failed = false;
  for (T i = 0; i < T(nthreads); ++i) {
    if(data_h[i] != out(i)) {
      std::cerr << i << ": " << data_h[i] << " != " << out(i) << std::endl;
      failed = true;
    }
  }

  if (failed) std::abort();

  HETEROGENEOUS_SAFE_CALL(cudaFree(data_d));
  delete[] data_h;
}

#define TESTS(T)				\
						\
  __global__ void store_##T(T* p) {			\
    T& data = p[threadIdx.x];				\
    cuda::atomic<T> x;					\
    x.store(data + 1, cuda::memory_order_relaxed);	\
    data = x.load(cuda::memory_order_relaxed);		\
  }							\
							\
  __global__ void compare_exchange_weak_##T(T* p) {			\
    T& data = p[threadIdx.x];						\
    cuda::atomic<T> x = data;						\
    T old = T(7);							\
    x.compare_exchange_weak(old, T(42), cuda::memory_order_relaxed);	\
    data = x.load(cuda::memory_order_relaxed);				\
  }									\
									\
  __global__ void compare_exchange_strong_##T(T* p) {			\
    T& data = p[threadIdx.x];						\
    cuda::atomic<T> x = data;						\
    T old = T(7);							\
    x.compare_exchange_strong(old, T(42), cuda::memory_order_relaxed);	\
    data = x.load(cuda::memory_order_relaxed);				\
  }									\
									\
  __global__ void exchange_##T(T* p) {					\
    T& data = p[threadIdx.x];						\
    cuda::atomic<T> x = data;						\
    T out = x.exchange(T(1), cuda::memory_order_relaxed);		\
    data = out + x.load(cuda::memory_order_relaxed);			\
  }									\
									\
  __global__ void fetch_add_##T(T* p) {					\
    T& data = p[threadIdx.x];						\
    cuda::atomic<T> x = data;						\
    x.fetch_add(T(1), cuda::memory_order_relaxed);			\
    data = x.load(cuda::memory_order_relaxed);				\
  }									\
									\
  __global__ void fetch_sub_##T(T* p) {					\
    T& data = p[threadIdx.x];						\
    cuda::atomic<T> x = data;						\
    x.fetch_sub(T(1), cuda::memory_order_relaxed);			\
    data = x.load(cuda::memory_order_relaxed);				\
  }									\
									\
  __global__ void fetch_and_##T(T* p) {					\
    T& data = p[threadIdx.x];						\
    cuda::atomic<T> x = data;						\
    x.fetch_and(T(1), cuda::memory_order_relaxed);			\
    data = x.load(cuda::memory_order_relaxed);				\
  }									\
									\
  __global__ void fetch_or_##T(T* p) {					\
    T& data = p[threadIdx.x];						\
    cuda::atomic<T> x = data;						\
    x.fetch_or(T(1), cuda::memory_order_relaxed);			\
    data = x.load(cuda::memory_order_relaxed);				\
  }									\
									\
  __global__ void fetch_xor_##T(T* p) {					\
    T& data = p[threadIdx.x];						\
    cuda::atomic<T> x = data;						\
    x.fetch_xor(T(1), cuda::memory_order_relaxed);			\
    data = x.load(cuda::memory_order_relaxed);				\
  }									\
									\
  __global__ void fetch_min_##T(T* p) {					\
    T& data = p[threadIdx.x];						\
    cuda::atomic<T> x = data;						\
    x.fetch_min(T(7), cuda::memory_order_relaxed);			\
    data = x.load(cuda::memory_order_relaxed);				\
  }									\
									\
  __global__ void fetch_max_##T(T* p) {					\
    T& data = p[threadIdx.x];						\
    cuda::atomic<T> x = data;						\
    x.fetch_max(T(7), cuda::memory_order_relaxed);			\
    data = x.load(cuda::memory_order_relaxed);				\
  }									\
  __host__ inline void tests_##T() {					\
    test([](T i) { return i; }, [](T i) -> T { return i + T(1); }, store_##T); \
    test([](T i) { return i; }, [](T i) -> T { return T(1) + i; }, exchange_##T); \
    test([](T i) { return i; }, [](T i) -> T { return i == T(7)? T(42) : i; }, compare_exchange_weak_##T); \
    test([](T i) { return i; }, [](T i) -> T { return i == T(7)? T(42) : i; }, compare_exchange_strong_##T); \
    test([](T i) { return i; }, [](T i) -> T { return i + T(1); }, fetch_add_##T); \
    test([](T i) { return i; }, [](T i) -> T { return i - T(1); }, fetch_sub_##T); \
    test([](T i) { return i; }, [](T i) -> T { return i & T(1); }, fetch_and_##T); \
    test([](T i) { return i; }, [](T i) -> T { return i | T(1); }, fetch_or_##T); \
    test([](T i) { return i; }, [](T i) -> T { return i ^ T(1); }, fetch_xor_##T); \
    test([](T i) { return i; }, [](T i) -> T { return std::min(i, T(7)); }, fetch_min_##T); \
    test([](T i) { return i; }, [](T i) -> T { return std::max(i, T(7)); }, fetch_max_##T); \
  } \


TESTS(uint8_t)
TESTS(uint16_t)
TESTS(uint32_t)
TESTS(uint64_t)
TESTS(int8_t)
TESTS(int16_t)
TESTS(int32_t)
TESTS(int64_t)

int main(int arg, char ** argv)
{

#ifndef __CUDA_ARCH__
  tests_uint8_t();
  tests_uint16_t();
  tests_uint32_t();
  tests_uint64_t();
  tests_int8_t();
  tests_int16_t();
  tests_int32_t();
  tests_int64_t();
#endif

  return 0;
}
