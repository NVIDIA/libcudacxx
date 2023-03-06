//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <complex>

#include <cuda/std/complex>
#include <cuda/std/cassert>

#include "test_macros.h"

template <class T, class U>
void test_construction() {
    const ::std::complex<U> only_real{static_cast<U>(42.0), 0 };
    const ::std::complex<U> only_imag{0, static_cast<U>(42.0)};
    const ::std::complex<U> real_imag{static_cast<U>(42.0), static_cast<U>(1337.0)};

    const ::cuda::std::complex<T> from_only_real{only_real};
    const ::cuda::std::complex<T> from_only_imag{only_imag};
    const ::cuda::std::complex<T> from_real_imag{real_imag};

    assert(from_only_real.real() == static_cast<T>(42.0));
    assert(from_only_real.imag() == 0);
    assert(from_only_imag.real() == 0);
    assert(from_only_imag.imag() == static_cast<T>(42.0));
    assert(from_real_imag.real() == static_cast<T>(42.0));
    assert(from_real_imag.imag() == static_cast<T>(1337.0));
}

template <class T, class U>
void test_assignment() {
    ::cuda::std::complex<T> from_only_real{static_cast<T>(-1.0), static_cast<T>(1.0)};
    ::cuda::std::complex<T> from_only_imag{static_cast<T>(-1.0), static_cast<T>(1.0)};
    ::cuda::std::complex<T> from_real_imag{static_cast<T>(-1.0), static_cast<T>(1.0)};

    const ::std::complex<U> only_real{static_cast<U>(42.0), 0 };
    const ::std::complex<U> only_imag{0, static_cast<U>(42.0)};
    const ::std::complex<U> real_imag{static_cast<U>(42.0), static_cast<U>(1337.0)};

    from_only_real = only_real;
    from_only_imag = only_imag;
    from_real_imag = real_imag;

    assert(from_only_real.real() == static_cast<T>(42.0));
    assert(from_only_real.imag() == 0);
    assert(from_only_imag.real() == 0);
    assert(from_only_imag.imag() == static_cast<T>(42.0));
    assert(from_real_imag.real() == static_cast<T>(42.0));
    assert(from_real_imag.imag() == static_cast<T>(1337.0));
}

template <class T>
void test_conversion() {
    const ::cuda::std::complex<T> only_real{static_cast<T>(42.0), 0 };
    const ::cuda::std::complex<T> only_imag{0, static_cast<T>(42.0)};
    const ::cuda::std::complex<T> real_imag{static_cast<T>(42.0), static_cast<T>(1337.0)};

    const ::std::complex<T> from_only_real{only_real};
    const ::std::complex<T> from_only_imag{only_imag};
    const ::std::complex<T> from_real_imag{real_imag};

    assert(from_only_real.real() == static_cast<T>(42.0));
    assert(from_only_real.imag() == 0);
    assert(from_only_imag.real() == 0);
    assert(from_only_imag.imag() == static_cast<T>(42.0));
    assert(from_real_imag.real() == static_cast<T>(42.0));
    assert(from_real_imag.imag() == static_cast<T>(1337.0));
}

template <class T, class U>
void test_comparison() {
    ::cuda::std::complex<T> input{static_cast<T>(-1.0), static_cast<T>(1.0)};

    const ::std::complex<U> not_equal_real{static_cast<T>(-1.0), 0};
    const ::std::complex<U> not_equal_imag{0, static_cast<T>(1.0)};
    const ::std::complex<U> equal{static_cast<T>(-1.0), static_cast<T>(1.0)};

    assert(!(input == not_equal_real));
    assert(!(input == not_equal_imag));
    assert(  input == equal);

    assert(  input != not_equal_real);
    assert(  input != not_equal_imag);
    assert(!(input != equal));
}

void test() {
    test_construction<float, float>();
    test_construction<double, float>();
    test_construction<double, double>();

    test_assignment<float, float>();
    test_assignment<double, float>();
    test_assignment<double, double>();

    test_conversion<float>();
    test_conversion<double>();

    test_comparison<float, float>();
    test_comparison<double, float>();
    test_comparison<double, double>();
}


int main(int arg, char ** argv)
{
#ifndef __CUDA_ARCH__
    test();
#endif // __CUDA_ARCH__
    return 0;
}