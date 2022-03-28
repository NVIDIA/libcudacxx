//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70
// UNSUPPORTED: !nvcc
// UNSUPPORTED: nvrtc

#include "utils.h"

/*
template <typename T, typename U, typename V>
__device__ __host__ __noinline__
void test_constructors(P p) {
      T* p = nullptr;
      cuda::access_property ap(cuda::access_property::persisting{});
      using P = cuda::access_property;
      // construct U from T
      cuda::annotated_ptr<U, P> q0{array0, ap};
      cuda::annotated_ptr<U, P> q1 = q0;
      cuda::annotated_ptr<U, P> q2{a2};
      // construct V from U
      cuda::annotated_ptr<V, P> q3 = q1;
      cuda::annotated_ptr<V, P> q4{q1};
      
      assert(q0 == q1);
      assert(q1 == q2);
      unused(q0);
      unused(q1);
      unused(q2);
      unused(q3);
      unused(q4);
}
*/


template<typename T, typename P>
__host__ __device__ __noinline__
void test_global_ctor() {
  // default ctor, cpy and cpy assignment
  cuda::annotated_ptr<T, P> def;
  def = def;
  cuda::annotated_ptr<T, P> other(def);

  // from ptr
  T* rp = nullptr;
  cuda::annotated_ptr<T, P> a(rp);

  // from ptr + prop
  P p;
  cuda::annotated_ptr<T, cuda::access_property> b(rp, p);

  // cpy ctor & asign to cv
  cuda::annotated_ptr<const T, P> c(def);
}

__host__ __device__ __noinline__
void test_global_ctors() {
  test_global_ctor<int, cuda::access_property::normal>();
  test_global_ctor<int, cuda::access_property::streaming>();
  test_global_ctor<int, cuda::access_property::persisting>();
  test_global_ctor<int, cuda::access_property::global>();
  test_global_ctor<int, cuda::access_property>();
}

int main(int argc, char ** argv)
{
  test_global_ctors();
  return 0;
}
