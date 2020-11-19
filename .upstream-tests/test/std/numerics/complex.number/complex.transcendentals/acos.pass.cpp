//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<class T>
//   complex<T>
//   acos(const complex<T>& x);

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ void
test(const cuda::std::complex<T>& c, cuda::std::complex<T> x)
{
    assert(acos(c) == x);
}

template <class T>
__host__ __device__ void
test()
{
    test(cuda::std::complex<T>(INFINITY, 1), cuda::std::complex<T>(0, -INFINITY));
}

__host__ __device__ void test_edges()
{
    const double pi = cuda::std::atan2(+0., -0.);
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        cuda::std::complex<double> r = acos(testcases[i]);
        if (testcases[i].real() == 0 && testcases[i].imag() == 0)
        {
            is_about(r.real(), pi/2);
            assert(r.imag() == 0);
            assert(cuda::std::signbit(testcases[i].imag()) != cuda::std::signbit(r.imag()));
        }
        else if (testcases[i].real() == 0 && cuda::std::isnan(testcases[i].imag()))
        {
            is_about(r.real(), pi/2);
            assert(cuda::std::isnan(r.imag()));
        }
        else if (cuda::std::isfinite(testcases[i].real()) && cuda::std::isinf(testcases[i].imag()))
        {
            is_about(r.real(), pi/2);
            assert(cuda::std::isinf(r.imag()));
            assert(cuda::std::signbit(testcases[i].imag()) != cuda::std::signbit(r.imag()));
        }
        else if (cuda::std::isfinite(testcases[i].real()) && testcases[i].real() != 0 && cuda::std::isnan(testcases[i].imag()))
        {
            assert(cuda::std::isnan(r.real()));
            assert(cuda::std::isnan(r.imag()));
        }
        else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() < 0 && cuda::std::isfinite(testcases[i].imag()))
        {
            is_about(r.real(), pi);
            assert(cuda::std::isinf(r.imag()));
            assert(cuda::std::signbit(testcases[i].imag()) != cuda::std::signbit(r.imag()));
        }
        else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() > 0 && cuda::std::isfinite(testcases[i].imag()))
        {
            assert(r.real() == 0);
            assert(!cuda::std::signbit(r.real()));
            assert(cuda::std::isinf(r.imag()));
            assert(cuda::std::signbit(testcases[i].imag()) != cuda::std::signbit(r.imag()));
        }
        else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() < 0 && cuda::std::isinf(testcases[i].imag()))
        {
            is_about(r.real(), 0.75 * pi);
            assert(cuda::std::isinf(r.imag()));
            assert(cuda::std::signbit(testcases[i].imag()) != cuda::std::signbit(r.imag()));
        }
        else if (cuda::std::isinf(testcases[i].real()) && testcases[i].real() > 0 && cuda::std::isinf(testcases[i].imag()))
        {
            is_about(r.real(), 0.25 * pi);
            assert(cuda::std::isinf(r.imag()));
            assert(cuda::std::signbit(testcases[i].imag()) != cuda::std::signbit(r.imag()));
        }
        else if (cuda::std::isinf(testcases[i].real()) && cuda::std::isnan(testcases[i].imag()))
        {
            assert(cuda::std::isnan(r.real()));
            assert(cuda::std::isinf(r.imag()));
        }
        else if (cuda::std::isnan(testcases[i].real()) && cuda::std::isfinite(testcases[i].imag()))
        {
            assert(cuda::std::isnan(r.real()));
            assert(cuda::std::isnan(r.imag()));
        }
        else if (cuda::std::isnan(testcases[i].real()) && cuda::std::isinf(testcases[i].imag()))
        {
            assert(cuda::std::isnan(r.real()));
            assert(cuda::std::isinf(r.imag()));
            assert(cuda::std::signbit(testcases[i].imag()) != cuda::std::signbit(r.imag()));
        }
        else if (cuda::std::isnan(testcases[i].real()) && cuda::std::isnan(testcases[i].imag()))
        {
            assert(cuda::std::isnan(r.real()));
            assert(cuda::std::isnan(r.imag()));
        }
        else if (!cuda::std::signbit(testcases[i].real()) && !cuda::std::signbit(testcases[i].imag()))
        {
            assert(!cuda::std::signbit(r.real()));
            assert( cuda::std::signbit(r.imag()));
        }
        else if (cuda::std::signbit(testcases[i].real()) && !cuda::std::signbit(testcases[i].imag()))
        {
            assert(!cuda::std::signbit(r.real()));
            assert( cuda::std::signbit(r.imag()));
        }
        else if (cuda::std::signbit(testcases[i].real()) && cuda::std::signbit(testcases[i].imag()))
        {
            assert(!cuda::std::signbit(r.real()));
            assert(!cuda::std::signbit(r.imag()));
        }
        else if (!cuda::std::signbit(testcases[i].real()) && cuda::std::signbit(testcases[i].imag()))
        {
            assert(!cuda::std::signbit(r.real()));
            assert(!cuda::std::signbit(r.imag()));
        }
    }
}

int main(int, char**)
{
    test<float>();
    test<double>();
// CUDA treats long double as double
//  test<long double>();
    test_edges();

  return 0;
}
