//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70
// UNSUPPORTED: c++98, c++03

#if defined(_LIBCUDACXX_COMPILER_MSVC)
#pragma warning(disable: 4505)
#endif

#include <cuda/annotated_ptr>
#include <cooperative_groups.h>

#ifndef __CUDA_ARCH__
#include <array>
#endif

#if defined(DEBUG)
    #define DPRINTF(...) { printf(__VA_ARGS__); }
#else
    #define DPRINTF(...) do {} while (false)
#endif

__device__ __host__
void assert_rt_wrap(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
#ifndef __CUDACC_RTC__
        printf("assert: %s %s %d\n", cudaGetErrorString(code), file, line);
#endif
        assert(code == cudaSuccess);
    }
}
#define assert_rt(ret) { assert_rt_wrap((ret), __FILE__, __LINE__); }


template <typename ... T>
__host__ __device__ constexpr bool unused(T...) {return true;}




/// Test read value
template <typename T, typename P>
static __host__ __device__
void read_value(T* ptr, P p, T expected) {
  cuda::annotated_ptr<T, P> ap{ptr, p};
  cuda::annotated_ptr<T, P> other{ap};

  assert(ap);
  // assert(ap == other);  // NOTE: not supported
  assert(*ap == expected);
  assert(ap[0] == expected);
  // assert(ap + 0 == ap);  // NOTE: not supported
  assert(ap.get() == ptr);
  assert((ap - ap) == nullptr);
  // TODO: why is this exposed?
  // assert(ap.__property() == p); 
}

__host__ __device__
void read_value_tests() {
  int x = 42;
  int* p = &x;
  read_value(p, cuda::access_property::persisting{}, x + 1);
}

__global__ void read_value_kernel() { read_value_tests(); }

void test_read_value() {
    // Host tests:
    read_value_tests();

    read_value_kernel<<<1, 1, 0, 0>>>();
    assert_rt(cudaStreamSynchronize(0));
}







template <typename In, typename Out, typename P>
__device__ __host__ __noinline__
void test_access(In* i, Out* o, size_t N) {
  cuda::annotated_ptr<In, P> ai{i};
  cuda::annotated_ptr<Out, P> ao{o};

  assert((i && (bool)ai) || (!i && !(bool)ai));
  assert((o && (bool)ao) || (!o && !(bool)ao));

  for (int n = 0; n < N; ++n) {
    assert(i[n] == n);
    assert(o[n] == 0);
    assert(&i[n] == &i[n]);
    assert(&o[n] == &o[n]);
    o[n] = i[n];
    assert(o[n] == n);
    o[n] = 0;
  }
}

template <typename In, typename Out, typename P>
__device__ __host__ __noinline__
void test_const_access(In* i, Out* o, size_t N) {
  const cuda::annotated_ptr<In, P> ai{i};
  const cuda::annotated_ptr<Out, P> ao{o};

  assert((i && (bool)ai) || (!i && !(bool)ai));
  assert((o && (bool)ao) || (!o && !(bool)ao));

  for (int n = 0; n < N; ++n) {
    assert(i[n] == n);
    assert(o[n] == 0);
    assert(&i[n] == &i[n]);
    assert(&o[n] == &o[n]);
    o[n] = i[n];
    assert(o[n] == n);
    o[n] = 0;
  }
}

__device__ __host__ __noinline__
void test_annotated_ptr_basic() {
  {
    test_constructors<int, int, int>();
    test_constructors<int, const int, const volatile int>();
    test_constructors<int, volatile int, const volatile int>();
    test_constructors<int, const volatile int, const volatile int>();

    using S = cuda::access_property::shared;
    #ifdef __CUDA_ARCH__
    __shared__ int q;
    cuda::annotated_ptr<int, S> s0{q};
    cuda::annotated_ptr<const int, S> s1{q};
    cuda::annotated_ptr<volatile int, S> s2{q};
    cuda::annotated_ptr<const volatile int, S> s3{q};
    #else
    cuda::annotated_ptr<int, S> s0{};
    cuda::annotated_ptr<const int, S> s1{};
    cuda::annotated_ptr<volatile int, S> s2{};
    cuda::annotated_ptr<const volatile int, S> s3{};
    #endif
    s3 = s1;
    s1 = s0;
    s2 = s1;
    unused(s0);
    unused(s1);
    unused(s2);
    unused(s3);

  }
  cuda::access_property ap(cuda::access_property::persisting{});
  static const size_t ARR_SZ = 1 << 10;

  {
    #ifndef __CUDA_ARCH__
    //Check on host the arrays through annotated_ptr ops
    using A = std::array<int, 3>;
    A a{3, 2, 1};
    cuda::annotated_ptr<A, cuda::access_property> p{&a, ap};
    assert(p->at(0) == 3);
    cuda::annotated_ptr<const A, cuda::access_property> q{&a, ap};
    assert(q->at(0) == 3);
    const A b{3, 2, 1};
    cuda::annotated_ptr<const A, cuda::access_property> r{&b, ap};
    assert(r->at(1) == 2);
    #endif
  }

  //Fill the arrays
  int* a0 = new int[ARR_SZ];
  int* a1 = new int[ARR_SZ];

  for (size_t i = 0; i < ARR_SZ; ++i) {
    a0[i] = static_cast<int>(i);
    a1[i] = 0;
  }
  using P = cuda::access_property::persisting;
  test_access<int, int, P>(a0, a1, ARR_SZ);
  test_access<const int, int, P>((const int*)a0, a1, ARR_SZ);
  test_access<const volatile int, int, P>((const volatile int*)a0, a1, ARR_SZ);
  test_access<const volatile int, volatile int, P>((const volatile int*)a0, (volatile int*)a1);

  test_const_access<int, int, P>(a0, a1, ARR_SZ);
  test_const_access<const int, int, P>((const int*)a0, a1, ARR_SZ);
  test_const_access<const volatile int, int, P>((const volatile int*)a0, a1, ARR_SZ);
  test_const_access<const volatile int, volatile int, P>((const volatile int*)a0, (volatile int*)a1, ARR_SZ);
  
  delete[] a0;
  delete[] a1;
}
