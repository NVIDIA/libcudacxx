//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<class T, class charT, class traits>
//   basic_istream<charT, traits>&
//   operator>>(basic_istream<charT, traits>& is, complex<T>& x);

#include <cuda/std/complex>
#include <cuda/std/sstream>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        cuda::std::istringstream is("5");
        cuda::std::complex<double> c;
        is >> c;
        assert(c == cuda::std::complex<double>(5, 0));
        assert(is.eof());
    }
    {
        cuda::std::istringstream is(" 5 ");
        cuda::std::complex<double> c;
        is >> c;
        assert(c == cuda::std::complex<double>(5, 0));
        assert(is.good());
    }
    {
        cuda::std::istringstream is(" 5, ");
        cuda::std::complex<double> c;
        is >> c;
        assert(c == cuda::std::complex<double>(5, 0));
        assert(is.good());
    }
    {
        cuda::std::istringstream is(" , 5, ");
        cuda::std::complex<double> c;
        is >> c;
        assert(c == cuda::std::complex<double>(0, 0));
        assert(is.fail());
    }
    {
        cuda::std::istringstream is("5.5 ");
        cuda::std::complex<double> c;
        is >> c;
        assert(c == cuda::std::complex<double>(5.5, 0));
        assert(is.good());
    }
    {
        cuda::std::istringstream is(" ( 5.5 ) ");
        cuda::std::complex<double> c;
        is >> c;
        assert(c == cuda::std::complex<double>(5.5, 0));
        assert(is.good());
    }
    {
        cuda::std::istringstream is("  5.5)");
        cuda::std::complex<double> c;
        is >> c;
        assert(c == cuda::std::complex<double>(5.5, 0));
        assert(is.good());
    }
    {
        cuda::std::istringstream is("(5.5 ");
        cuda::std::complex<double> c;
        is >> c;
        assert(c == cuda::std::complex<double>(0, 0));
        assert(is.fail());
    }
    {
        cuda::std::istringstream is("(5.5,");
        cuda::std::complex<double> c;
        is >> c;
        assert(c == cuda::std::complex<double>(0, 0));
        assert(is.fail());
    }
    {
        cuda::std::istringstream is("( -5.5 , -6.5 )");
        cuda::std::complex<double> c;
        is >> c;
        assert(c == cuda::std::complex<double>(-5.5, -6.5));
        assert(!is.eof());
    }
    {
        cuda::std::istringstream is("(-5.5,-6.5)");
        cuda::std::complex<double> c;
        is >> c;
        assert(c == cuda::std::complex<double>(-5.5, -6.5));
        assert(!is.eof());
    }

  return 0;
}
