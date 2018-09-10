/*

Copyright (c) 2018, NVIDIA Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#include <cstddef>
#include <cstdint>
#include <atomic>

// stay tuned for <algorithm>
template<class T> static constexpr T min(T a, T b) { return a < b ? a : b; }

struct node {
    struct ref {
        std::atomic<node*>  ptr = ATOMIC_VAR_INIT(nullptr);
        std::atomic_flag    once = ATOMIC_FLAG_INIT;
    } next[26];
    std::atomic<int> count = ATOMIC_VAR_INIT(0);
};
struct trie {
    std::atomic<node*> bump = ATOMIC_VAR_INIT(nullptr);
    node                     root;
    trie(node* ptr) : bump(ptr) { }
};

void process(const char* begin, const char* end, trie* t, unsigned const index, unsigned const range) {

    auto const size = end - begin;
    auto const stride = (size / range + 1);

    auto off = min(size, stride * index);
    auto const last = min(size, off + stride);

    auto const index_of = [](char c) -> int {
        if(c >= 'a' && c <= 'z') return c - 'a';
        if(c >= 'A' && c <= 'Z') return c - 'A';
        return -1;
    };

    for(char c = begin[off]; off < size && off != last && c != 0 && index_of(c) != -1; ++off, c = begin[off]);
    for(char c = begin[off]; off < size && off != last && c != 0 && index_of(c) == -1; ++off, c = begin[off]);

    node *const proot = &t->root, *n = proot;
    for(char c = begin[off]; ; ++off, c = begin[off]) {
        auto const index = off >= size ? -1 : index_of(c);
        if(index == -1) {
            if(n != proot) {
                n->count.fetch_add(1, std::memory_order_relaxed);
                n = proot;
            }
            //end of last word?
            if(off >= size || off > last)
                break;
            else
                continue;
        }
        auto& ptr = n->next[index].ptr;
        auto next = ptr.load(std::memory_order_acquire);
        if(next == nullptr) {
            auto& once = n->next[index].once;
            if(once.test_and_set()) {
                do {
                    next = ptr.load(std::memory_order_acquire);
                } while(next == nullptr);
            }
            else {
                next = ptr.load(std::memory_order_acquire);
                if(next == nullptr) {
                    next = t->bump.fetch_add(1, std::memory_order_relaxed);
                    ptr.store(next, std::memory_order_relaxed);
	            }
            }
        }
        n = next;
    }
}

#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <cassert>

void do_trie(std::string const& input, int threads) {
    
    std::vector<node> nodes(1<<20);
    trie t(nodes.data());

    auto const begin = std::chrono::steady_clock::now();
    std::atomic_signal_fence(std::memory_order_seq_cst);

    std::vector<std::thread> tv(threads);
    for(auto count = threads; count; --count)
        tv[count - 1] = std::thread([&, count]() {
            process(input.data(), input.data() + input.size(), &t, count - 1, threads);
        });
    for(auto& t : tv)
        t.join();

    std::atomic_signal_fence(std::memory_order_seq_cst);
    auto const end = std::chrono::steady_clock::now();
    auto const time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    auto const count = t.bump - nodes.data();
    std::cout << "Assembled " << count << " nodes on " << threads << " threads in " << time << "ms." << std::endl;
}

int main() {

    std::string input;

    char const* files[] = {
        "2600-0.txt", "2701-0.txt", "35-0.txt", "84-0.txt", "8800.txt",
      	"pg1727.txt", "pg55.txt", "pg6130.txt", "pg996.txt", "1342-0.txt"
    };

    std::size_t total = 0, cur = 0;
    for(auto* ptr : files) {
        std::ifstream in(ptr);
        in.seekg(0, std::ios_base::end);
        total += in.tellg();
    }
    input.resize(total);
    for(auto* ptr : files) {
        std::ifstream in(ptr);
        in.seekg(0, std::ios_base::end);
        auto const pos = in.tellg();
        in.seekg(0, std::ios_base::beg);
        in.read((char*)input.data() + cur, pos);
        cur += pos;
    }

    do_trie(input, 1);
    do_trie(input, 1);
    do_trie(input, std::thread::hardware_concurrency());
    do_trie(input, std::thread::hardware_concurrency());

    return 0;
}
