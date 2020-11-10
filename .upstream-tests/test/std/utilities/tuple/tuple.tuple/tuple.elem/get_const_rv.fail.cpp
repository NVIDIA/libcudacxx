//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <size_t I, class... Types>
//   const typename tuple_element<I, tuple<Types...> >::type&&
//   get(const tuple<Types...>&& t);

// UNSUPPORTED: c++98, c++03 
// UNSUPPORTED: nvrtc

#include <cuda/std/tuple>

template <class T> __host__ __device__ void cref(T const&) {}
template <class T> __host__ __device__ void cref(T const&&) = delete;

cuda::std::tuple<int> __host__ __device__ const tup4() { return cuda::std::make_tuple(4); }

int main(int, char**)
{
    // LWG2485: tuple should not open a hole in the type system, get() should
    // imitate [expr.ref]'s rules for accessing data members
    {
        cref(cuda::std::get<0>(tup4()));  // expected-error {{call to deleted function 'cref'}}
    }

  return 0;
}
