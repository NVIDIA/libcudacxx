//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/unordered_set>
// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: libcpp-no-deduction-guides
// XFAIL: clang-6, apple-clang-9.0, apple-clang-9.1, apple-clang-10.0.0

// template<class InputIterator,
//        class Hash = hash<iter-value-type<InputIterator>>,
//        class Pred = equal_to<iter-value-type<InputIterator>>,
//        class Allocator = allocator<iter-value-type<InputIterator>>>
// unordered_set(InputIterator, InputIterator, typename see below::size_type = see below,
//               Hash = Hash(), Pred = Pred(), Allocator = Allocator())
//   -> unordered_set<iter-value-type<InputIterator>,
//                    Hash, Pred, Allocator>;
//
// template<class T, class Hash = hash<T>,
//        class Pred = equal_to<T>, class Allocator = allocator<T>>
// unordered_set(initializer_list<T>, typename see below::size_type = see below,
//               Hash = Hash(), Pred = Pred(), Allocator = Allocator())
//   -> unordered_set<T, Hash, Pred, Allocator>;
//
// template<class InputIterator, class Allocator>
// unordered_set(InputIterator, InputIterator, typename see below::size_type, Allocator)
//   -> unordered_set<iter-value-type<InputIterator>,
//                    hash<iter-value-type<InputIterator>>,
//                    equal_to<iter-value-type<InputIterator>>,
//                    Allocator>;
//
// template<class InputIterator, class Hash, class Allocator>
// unordered_set(InputIterator, InputIterator, typename see below::size_type,
//               Hash, Allocator)
//   -> unordered_set<iter-value-type<InputIterator>, Hash,
//                    equal_to<iter-value-type<InputIterator>>,
//                    Allocator>;
//
// template<class T, class Allocator>
// unordered_set(initializer_list<T>, typename see below::size_type, Allocator)
//   -> unordered_set<T, hash<T>, equal_to<T>, Allocator>;
//
// template<class T, class Hash, class Allocator>
// unordered_set(initializer_list<T>, typename see below::size_type, Hash, Allocator)
//   -> unordered_set<T, Hash, equal_to<T>, Allocator>;

#include <cuda/std/functional>
#include <cuda/std/unordered_set>

int main(int, char**)
{
    {
        // cannot deduce Key from nothing
        cuda::std::unordered_set s;
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'unordered_set'}}
    }
    {
        // cannot deduce Key from just (Size)
        cuda::std::unordered_set s(42);
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'unordered_set'}}
    }
    {
        // cannot deduce Key from just (Size, Hash)
        cuda::std::unordered_set s(42, cuda::std::hash<int>());
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'unordered_set'}}
    }
    {
        // cannot deduce Key from just (Size, Hash, Pred)
        cuda::std::unordered_set s(42, cuda::std::hash<int>(), cuda::std::equal_to<>());
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'unordered_set'}}
    }
    {
        // cannot deduce Key from just (Size, Hash, Pred, Allocator)
        cuda::std::unordered_set s(42, cuda::std::hash<int>(), cuda::std::equal_to<>(), cuda::std::allocator<int>());
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'unordered_set'}}
    }
    {
        // cannot deduce Key from just (Allocator)
        cuda::std::unordered_set s(cuda::std::allocator<int>{});
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'unordered_set'}}
    }
    {
        // cannot deduce Key from just (Size, Allocator)
        cuda::std::unordered_set s(42, cuda::std::allocator<int>());
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'unordered_set'}}
    }
    {
        // cannot deduce Key from just (Size, Hash, Allocator)
        cuda::std::unordered_set s(42, cuda::std::hash<short>(), cuda::std::allocator<int>());
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'unordered_set'}}
    }

    return 0;
}
