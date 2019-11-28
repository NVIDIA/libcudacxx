//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// reference_wrapper

// reference_wrapper& operator=(const reference_wrapper<T>& x);

#include <cuda/std/functional>
#include <cuda/std/cassert>

class functor1
{
};

template <class T>
__host__ __device__
void
test(T& t)
{
    cuda::std::reference_wrapper<T> r(t);
    T t2 = t;
    cuda::std::reference_wrapper<T> r2(t2);
    r2 = r;
    assert(&r2.get() == &t);
}

__host__ __device__
void f() {}
__host__ __device__
void g() {}

__host__ __device__
void
test_function()
{
    cuda::std::reference_wrapper<void ()> r(f);
    cuda::std::reference_wrapper<void ()> r2(g);
    r2 = r;
    assert(&r2.get() == &f);
}

int main(int, char**)
{
    void (*fp)() = f;
    test(fp);
    test_function();
    functor1 f1;
    test(f1);
    int i = 0;
    test(i);
    const int j = 0;
    test(j);

  return 0;
}
