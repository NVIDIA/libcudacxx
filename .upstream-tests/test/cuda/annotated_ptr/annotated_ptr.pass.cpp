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
// UNSUPPORTED: c++98, c++03
// XFAIL: nvcc

#if 0

#include "annotated_ptr.h"

static_assert(sizeof(cuda::annotated_ptr<int, cuda::access_property::global>) == sizeof(uintptr_t),
	      "annotated_ptr<T, global> must be pointer size");
static_assert(sizeof(cuda::annotated_ptr<char, cuda::access_property::global>) == sizeof(uintptr_t),
	      "annotated_ptr<T, global> must be pointer size");
static_assert(sizeof(cuda::annotated_ptr<uintptr_t, cuda::access_property::global>) == sizeof(uintptr_t),
	      "annotated_ptr<T, global> must be pointer size");
static_assert(sizeof(cuda::annotated_ptr<uintptr_t, cuda::access_property::persisting>) == sizeof(uintptr_t),
	      "annotated_ptr<T, global> must be pointer size");
static_assert(sizeof(cuda::annotated_ptr<uintptr_t, cuda::access_property::normal>) == sizeof(uintptr_t),
	      "annotated_ptr<T, global> must be pointer size");
static_assert(sizeof(cuda::annotated_ptr<uintptr_t, cuda::access_property::streaming>) == sizeof(uintptr_t),
	      "annotated_ptr<T, global> must be pointer size");

static_assert(sizeof(cuda::annotated_ptr<uintptr_t, cuda::access_property>) == 2*sizeof(uintptr_t),
	      "annotated_ptr<T,access_property> must be 2 * pointer size");

// NOTE: we could make these smaller in the future (e.g. 32-bit) but that would be an ABI breaking change:
static_assert(sizeof(cuda::annotated_ptr<int, cuda::access_property::shared>) == sizeof(uintptr_t),
	      "annotated_ptr<T, shared> must be pointer size");
static_assert(sizeof(cuda::annotated_ptr<char, cuda::access_property::shared>) == sizeof(uintptr_t),
	      "annotated_ptr<T, shared> must be pointer size");
static_assert(sizeof(cuda::annotated_ptr<uintptr_t, cuda::access_property::shared>) == sizeof(uintptr_t),
	      "annotated_ptr<T, shared> must be pointer size");


static_assert(alignof(cuda::annotated_ptr<int, cuda::access_property::global>) == alignof(int*),
	      "annotated_ptr must align with int*");
static_assert(alignof(cuda::annotated_ptr<int, cuda::access_property::persisting>) == alignof(int*),
	      "annotated_ptr must align with int*");
static_assert(alignof(cuda::annotated_ptr<int, cuda::access_property::normal>) == alignof(int*),
	      "annotated_ptr must align with int*");
static_assert(alignof(cuda::annotated_ptr<int, cuda::access_property::streaming>) == alignof(int*),
	      "annotated_ptr must align with int*");
static_assert(alignof(cuda::annotated_ptr<int, cuda::access_property>) == alignof(int*),
	      "annotated_ptr must align with int*");

// NOTE: we could lower the alignment in the future but that would be an ABI breaking change:
static_assert(alignof(cuda::annotated_ptr<int, cuda::access_property::shared>) == alignof(int*),
	      "annotated_ptr must align with int*");


int main(int argc, char ** argv)
{
  test_read_value();
  /*test_annotated_ptr_basic();
    test_annotated_ptr_launch_kernel();
    test_annotated_ptr_functions();
  */
    return 0;
}

#else
int main() { return 0; }
#endif
