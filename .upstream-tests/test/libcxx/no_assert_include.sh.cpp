// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that none of the standard C++ headers implicitly include cassert or
// assert.h (because assert() is implemented as a macro).

// RUN: %compile -fsyntax-only

// Prevent <ext/hash_map> from generating deprecated warnings for this test.
#if defined(__DEPRECATED)
#undef __DEPRECATED
#endif

// Top level headers
#include <cuda/std/algorithm>
#include <cuda/std/any>
#include <cuda/std/array>
#ifndef _LIBCUDACXX_HAS_NO_THREADS
#include <cuda/std/atomic>
#endif
#include <cuda/std/bit>
#include <cuda/std/bitset>
#include <ccomplex>
#include <cuda/std/cctype>
#include <cuda/std/cerrno>
#include <cuda/std/cfenv>
#include <cuda/std/cfloat>
#include <cuda/std/charconv>
#include <cuda/std/chrono>
#include <cuda/std/cinttypes>
#include <ciso646>
#include <cuda/std/climits>
#include <cuda/std/clocale>
#include <cuda/std/cmath>
#include <cuda/std/codecvt>
#include <cuda/std/compare>
#include <cuda/std/complex>
#include <complex.h>
#include <cuda/std/condition_variable>
#include <cuda/std/csetjmp>
#include <cuda/std/csignal>
#include <cuda/std/cstdarg>
#include <cstdbool>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/cstdio>
#include <cuda/std/cstdlib>
#include <cuda/std/cstring>
#include <ctgmath>
#include <cuda/std/ctime>
#include <ctype.h>
#include <cuda/std/cwchar>
#include <cuda/std/cwctype>
#include <cuda/std/deque>
#include <errno.h>
#include <cuda/std/exception>
#include <cuda/std/execution>
#include <fenv.h>
#include <cuda/std/filesystem>
#include <float.h>
#include <cuda/std/forward_list>
#include <cuda/std/fstream>
#include <cuda/std/functional>
#ifndef _LIBCUDACXX_HAS_NO_THREADS
#include <cuda/std/future>
#endif
#include <cuda/std/initializer_list>
#include <inttypes.h>
#include <cuda/std/iomanip>
#include <cuda/std/ios>
#include <cuda/std/iosfwd>
#include <cuda/std/iostream>
#include <cuda/std/istream>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <limits.h>
#include <cuda/std/list>
#include <cuda/std/locale>
#include <locale.h>
#include <cuda/std/map>
#include <math.h>
#include <cuda/std/memory>
#ifndef _LIBCUDACXX_HAS_NO_THREADS
#include <cuda/std/mutex>
#endif
#include <cuda/std/new>
#include <cuda/std/numeric>
#include <cuda/std/optional>
#include <cuda/std/ostream>
#include <cuda/std/queue>
#include <cuda/std/random>
#include <cuda/std/ratio>
#include <cuda/std/regex>
#include <cuda/std/scoped_allocator>
#include <cuda/std/set>
#include <setjmp.h>
#ifndef _LIBCUDACXX_HAS_NO_THREADS
#include <cuda/std/shared_mutex>
#endif
#include <cuda/std/span>
#include <cuda/std/sstream>
#include <cuda/std/stack>
#include <stdbool.h>
#include <stddef.h>
#include <cuda/std/stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda/std/streambuf>
#include <cuda/std/string>
#include <string.h>
#include <cuda/std/string_view>
#include <cuda/std/strstream>
#include <cuda/std/system_error>
#include <tgmath.h>
#ifndef _LIBCUDACXX_HAS_NO_THREADS
#include <cuda/std/thread>
#endif
#include <cuda/std/tuple>
#include <cuda/std/typeindex>
#include <cuda/std/typeinfo>
#include <cuda/std/type_traits>
#include <cuda/std/unordered_map>
#include <cuda/std/unordered_set>
#include <cuda/std/utility>
#include <cuda/std/valarray>
#include <cuda/std/variant>
#include <cuda/std/vector>
#include <cuda/std/version>
#include <wchar.h>
#include <wctype.h>

// experimental headers
#if __cplusplus >= 201103L
#include <experimental/algorithm>
#if defined(__cpp_coroutines)
#include <experimental/coroutine>
#endif
#include <experimental/deque>
#include <experimental/filesystem>
#include <experimental/forward_list>
#include <experimental/functional>
#include <experimental/iterator>
#include <experimental/list>
#include <experimental/map>
#include <experimental/memory_resource>
#include <experimental/propagate_const>
#include <experimental/regex>
#include <experimental/simd>
#include <experimental/set>
#include <experimental/string>
#include <experimental/type_traits>
#include <experimental/unordered_map>
#include <experimental/unordered_set>
#include <experimental/utility>
#include <experimental/vector>
#endif // __cplusplus >= 201103L

// extended headers
#include <ext/hash_map>
#include <ext/hash_set>

#ifdef assert
#error "Do not include cassert or assert.h in standard header files"
#endif
