#include <cassert>



#include <gpu/cstdint>
#include <gpu/cstddef>
#include <gpu/atomic>

// preview on GitHub, monthly release at head
//#include <gpu/mutex>

namespace cuda { namespace std {
    struct mutex { 
        __host__ __device__ bool try_lock() { return true; } 
        __host__ __device__ void lock() { } 
        __host__ __device__ void unlock() { } 
    };
}}

// stay tuned for <algorithm>
template<class T> static constexpr T min(T a, T b) { return a < b ? a : b; }

struct node {
    struct ref {
        cuda::std::atomic<node*> ptr = ATOMIC_VAR_INIT(nullptr);
        cuda::std::mutex         lock;
    };
    ref                    next[26];
    cuda::std::atomic<int> count = ATOMIC_VAR_INIT(0);
};
struct trie {
    cuda::std::atomic<node*> bump = ATOMIC_VAR_INIT(nullptr);
    node                     root;
    __host__ __device__ trie(node* ptr) : bump(ptr) { }
};

__host__ __device__ void process(const char* begin, const char* end, trie* t, unsigned const index, unsigned const range) {

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
                n->count.fetch_add(1, cuda::std::memory_order_relaxed);
                n = proot;
            }
            //end of last word?
            if(off >= size || off > last)
                break;
            else
                continue;
        }
        auto& ptr = n->next[index].ptr;
        auto next = ptr.load(cuda::std::memory_order_acquire);
        if(next == nullptr) {
            auto& lock = n->next[index].lock;
            if(!lock.try_lock()) {
                do {
                    next = ptr.load(cuda::std::memory_order_acquire);
                } while(next == nullptr);
            }
            else {
                next = ptr.load(cuda::std::memory_order_acquire);
                if(next == nullptr) {
                    next = t->bump.fetch_add(1, cuda::std::memory_order_relaxed);
                    ptr.store(next, cuda::std::memory_order_relaxed);
                    lock.unlock();
	        }
	        else lock.unlock();
            }
        }
        n = next;
    }
}

__global__ __launch_bounds__(1024, 2) void call_process(const char* begin, const char* end, trie* t) {
    auto const index = blockDim.x * blockIdx.x + threadIdx.x;
    auto const range = gridDim.x * blockDim.x;
    process(begin, end, t, index, range);
}

#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <chrono>
#include <thread>

template <class T>
struct managed_allocator {
  typedef T value_type;
  managed_allocator() = default;
  template <class U> constexpr managed_allocator(const managed_allocator<U>&) noexcept {}
  [[nodiscard]] T* allocate(std::size_t n) {
    assert(n <= std::size_t(-1) / sizeof(T));
    void* out = nullptr;
    auto const ret = cudaMallocManaged(&out, n*sizeof(T));
    assert(ret == cudaSuccess);
    if(auto p = static_cast<T*>(out)) return p;
    return nullptr;
  }
  void deallocate(T* p, std::size_t) noexcept { 
      cudaFree(p); 
  }
};
template<class T, class... Args>
T* make_(Args &&... args) {
    managed_allocator<T> ma;
    return new (ma.allocate(1)) T(std::forward<Args>(args)...);
}

using string = std::basic_string<char, std::char_traits<char>, managed_allocator<char>>;
using vector = std::vector<node, managed_allocator<node>>;

void do_trie(string* input, vector* nodes, bool use_gpu, int blocks, int threads) {
    
    cudaMemset(nodes->data(), 0, nodes->size() * sizeof(node));

    trie* const t = make_<trie>(nodes->data());

    auto const begin = std::chrono::steady_clock::now();
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if(use_gpu) {
        call_process<<<blocks,threads>>>(input->data(), input->data() + input->size(), t);
        auto const ret = cudaDeviceSynchronize();
	assert(ret == cudaSuccess);
    }
    else {
        assert(blocks == 1);
        std::vector<std::thread> tv(threads);
        for(auto count = threads; count; --count)
            tv[count - 1] = std::thread([&, count]() {
                process(input->data(), input->data() + input->size(), t, count - 1, threads);
            });
        for(auto& t : tv)
            t.join();
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);
    auto const end = std::chrono::steady_clock::now();
    auto const time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    auto const count = t->bump - nodes->data();
    std::cout << "Assembled " << count << " nodes on " << blocks << "x" << threads << " " << (use_gpu ? "gpu" : "cpu") << " threads in " << time << "ms." << std::endl;
}

int main() {

    string* input = make_<string>();
    vector* nodes = make_<vector>(1<<20);

    char const* files[10] = { "divine_comedy.txt",
                              "frankenstein.txt",
                              "iliad.txt",  
                              "moby_dick.txt",  
                              "odyssey.txt",  
                              "oz.txt",  
                              "quixote.txt",  
                              "time_machine.txt",
                              "war_and_peace.txt",
                              "quixote.txt" };

    std::size_t total = 0, cur = 0;
    for(auto* ptr : files) {
        std::ifstream in(ptr);
        in.seekg(0, std::ios_base::end);
        total += in.tellg();
    }
    input->resize(total);
    for(auto* ptr : files) {
        std::ifstream in(ptr);
        in.seekg(0, std::ios_base::end);
        auto const pos = in.tellg();
        in.seekg(0, std::ios_base::beg);
        in.read((char*)input->data() + cur, pos);
        cur += pos;
    }

    do_trie(input, nodes, false, 1, 1);
    do_trie(input, nodes, false, 1, 1);
    do_trie(input, nodes, false, 1, std::thread::hardware_concurrency());
    do_trie(input, nodes, false, 1, std::thread::hardware_concurrency());

    if(cudaSetDevice(0) != cudaSuccess) return 1;
    cudaDeviceProp deviceProp;
    if(cudaGetDeviceProperties(&deviceProp, 0) != cudaSuccess) return 2;

    do_trie(input, nodes, true, deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor >> 10, 1<<10);
    do_trie(input, nodes, true, deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor >> 10, 1<<10);

    return 0;
}

