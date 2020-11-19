//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// test cases

#ifndef CASES_H
#define CASES_H

#include <cuda/std/complex>
#include <cuda/std/cassert>

#ifdef __CUDA_ARCH__
__constant__
#else
const
#endif
cuda::std::complex<double> testcases[] =
{
    cuda::std::complex<double>( 1.e-6,  1.e-6),
    cuda::std::complex<double>(-1.e-6,  1.e-6),
    cuda::std::complex<double>(-1.e-6, -1.e-6),
    cuda::std::complex<double>( 1.e-6, -1.e-6),

    cuda::std::complex<double>( 1.e+6,  1.e-6),
    cuda::std::complex<double>(-1.e+6,  1.e-6),
    cuda::std::complex<double>(-1.e+6, -1.e-6),
    cuda::std::complex<double>( 1.e+6, -1.e-6),

    cuda::std::complex<double>( 1.e-6,  1.e+6),
    cuda::std::complex<double>(-1.e-6,  1.e+6),
    cuda::std::complex<double>(-1.e-6, -1.e+6),
    cuda::std::complex<double>( 1.e-6, -1.e+6),

    cuda::std::complex<double>( 1.e+6,  1.e+6),
    cuda::std::complex<double>(-1.e+6,  1.e+6),
    cuda::std::complex<double>(-1.e+6, -1.e+6),
    cuda::std::complex<double>( 1.e+6, -1.e+6),

    cuda::std::complex<double>(NAN, NAN),
    cuda::std::complex<double>(-INFINITY, NAN),
    cuda::std::complex<double>(-2, NAN),
    cuda::std::complex<double>(-1, NAN),
    cuda::std::complex<double>(-0.5, NAN),
    cuda::std::complex<double>(-0., NAN),
    cuda::std::complex<double>(+0., NAN),
    cuda::std::complex<double>(0.5, NAN),
    cuda::std::complex<double>(1, NAN),
    cuda::std::complex<double>(2, NAN),
    cuda::std::complex<double>(INFINITY, NAN),

    cuda::std::complex<double>(NAN, -INFINITY),
    cuda::std::complex<double>(-INFINITY, -INFINITY),
    cuda::std::complex<double>(-2, -INFINITY),
    cuda::std::complex<double>(-1, -INFINITY),
    cuda::std::complex<double>(-0.5, -INFINITY),
    cuda::std::complex<double>(-0., -INFINITY),
    cuda::std::complex<double>(+0., -INFINITY),
    cuda::std::complex<double>(0.5, -INFINITY),
    cuda::std::complex<double>(1, -INFINITY),
    cuda::std::complex<double>(2, -INFINITY),
    cuda::std::complex<double>(INFINITY, -INFINITY),

    cuda::std::complex<double>(NAN, -2),
    cuda::std::complex<double>(-INFINITY, -2),
    cuda::std::complex<double>(-2, -2),
    cuda::std::complex<double>(-1, -2),
    cuda::std::complex<double>(-0.5, -2),
    cuda::std::complex<double>(-0., -2),
    cuda::std::complex<double>(+0., -2),
    cuda::std::complex<double>(0.5, -2),
    cuda::std::complex<double>(1, -2),
    cuda::std::complex<double>(2, -2),
    cuda::std::complex<double>(INFINITY, -2),

    cuda::std::complex<double>(NAN, -1),
    cuda::std::complex<double>(-INFINITY, -1),
    cuda::std::complex<double>(-2, -1),
    cuda::std::complex<double>(-1, -1),
    cuda::std::complex<double>(-0.5, -1),
    cuda::std::complex<double>(-0., -1),
    cuda::std::complex<double>(+0., -1),
    cuda::std::complex<double>(0.5, -1),
    cuda::std::complex<double>(1, -1),
    cuda::std::complex<double>(2, -1),
    cuda::std::complex<double>(INFINITY, -1),

    cuda::std::complex<double>(NAN, -0.5),
    cuda::std::complex<double>(-INFINITY, -0.5),
    cuda::std::complex<double>(-2, -0.5),
    cuda::std::complex<double>(-1, -0.5),
    cuda::std::complex<double>(-0.5, -0.5),
    cuda::std::complex<double>(-0., -0.5),
    cuda::std::complex<double>(+0., -0.5),
    cuda::std::complex<double>(0.5, -0.5),
    cuda::std::complex<double>(1, -0.5),
    cuda::std::complex<double>(2, -0.5),
    cuda::std::complex<double>(INFINITY, -0.5),

    cuda::std::complex<double>(NAN, -0.),
    cuda::std::complex<double>(-INFINITY, -0.),
    cuda::std::complex<double>(-2, -0.),
    cuda::std::complex<double>(-1, -0.),
    cuda::std::complex<double>(-0.5, -0.),
    cuda::std::complex<double>(-0., -0.),
    cuda::std::complex<double>(+0., -0.),
    cuda::std::complex<double>(0.5, -0.),
    cuda::std::complex<double>(1, -0.),
    cuda::std::complex<double>(2, -0.),
    cuda::std::complex<double>(INFINITY, -0.),

    cuda::std::complex<double>(NAN, +0.),
    cuda::std::complex<double>(-INFINITY, +0.),
    cuda::std::complex<double>(-2, +0.),
    cuda::std::complex<double>(-1, +0.),
    cuda::std::complex<double>(-0.5, +0.),
    cuda::std::complex<double>(-0., +0.),
    cuda::std::complex<double>(+0., +0.),
    cuda::std::complex<double>(0.5, +0.),
    cuda::std::complex<double>(1, +0.),
    cuda::std::complex<double>(2, +0.),
    cuda::std::complex<double>(INFINITY, +0.),

    cuda::std::complex<double>(NAN, 0.5),
    cuda::std::complex<double>(-INFINITY, 0.5),
    cuda::std::complex<double>(-2, 0.5),
    cuda::std::complex<double>(-1, 0.5),
    cuda::std::complex<double>(-0.5, 0.5),
    cuda::std::complex<double>(-0., 0.5),
    cuda::std::complex<double>(+0., 0.5),
    cuda::std::complex<double>(0.5, 0.5),
    cuda::std::complex<double>(1, 0.5),
    cuda::std::complex<double>(2, 0.5),
    cuda::std::complex<double>(INFINITY, 0.5),

    cuda::std::complex<double>(NAN, 1),
    cuda::std::complex<double>(-INFINITY, 1),
    cuda::std::complex<double>(-2, 1),
    cuda::std::complex<double>(-1, 1),
    cuda::std::complex<double>(-0.5, 1),
    cuda::std::complex<double>(-0., 1),
    cuda::std::complex<double>(+0., 1),
    cuda::std::complex<double>(0.5, 1),
    cuda::std::complex<double>(1, 1),
    cuda::std::complex<double>(2, 1),
    cuda::std::complex<double>(INFINITY, 1),

    cuda::std::complex<double>(NAN, 2),
    cuda::std::complex<double>(-INFINITY, 2),
    cuda::std::complex<double>(-2, 2),
    cuda::std::complex<double>(-1, 2),
    cuda::std::complex<double>(-0.5, 2),
    cuda::std::complex<double>(-0., 2),
    cuda::std::complex<double>(+0., 2),
    cuda::std::complex<double>(0.5, 2),
    cuda::std::complex<double>(1, 2),
    cuda::std::complex<double>(2, 2),
    cuda::std::complex<double>(INFINITY, 2),

    cuda::std::complex<double>(NAN, INFINITY),
    cuda::std::complex<double>(-INFINITY, INFINITY),
    cuda::std::complex<double>(-2, INFINITY),
    cuda::std::complex<double>(-1, INFINITY),
    cuda::std::complex<double>(-0.5, INFINITY),
    cuda::std::complex<double>(-0., INFINITY),
    cuda::std::complex<double>(+0., INFINITY),
    cuda::std::complex<double>(0.5, INFINITY),
    cuda::std::complex<double>(1, INFINITY),
    cuda::std::complex<double>(2, INFINITY),
    cuda::std::complex<double>(INFINITY, INFINITY)
};

enum {zero, non_zero, inf, NaN, non_zero_nan};

template <class T>
__host__ __device__ int
classify(const cuda::std::complex<T>& x)
{
    if (x == cuda::std::complex<T>())
        return zero;
    if (std::isinf(x.real()) || std::isinf(x.imag()))
        return inf;
    if (std::isnan(x.real()) && std::isnan(x.imag()))
        return NaN;
    if (std::isnan(x.real()))
    {
        if (x.imag() == T(0))
            return NaN;
        return non_zero_nan;
    }
    if (std::isnan(x.imag()))
    {
        if (x.real() == T(0))
            return NaN;
        return non_zero_nan;
    }
    return non_zero;
}

inline
__host__ __device__ int
classify(double x)
{
    if (x == 0)
        return zero;
    if (std::isinf(x))
        return inf;
    if (std::isnan(x))
        return NaN;
    return non_zero;
}

__host__ __device__ void is_about(float x, float y)
{
    assert(std::abs((x-y)/(x+y)) < 1.e-6);
}

__host__ __device__ void is_about(double x, double y)
{
    assert(std::abs((x-y)/(x+y)) < 1.e-14);
}

// CUDA treats long double as double
/*
__host__ __device__ void is_about(long double x, long double y)
{
    assert(std::abs((x-y)/(x+y)) < 1.e-14);
}
*/
#endif  // CASES_H
