#include <cassert>
struct type { /* a regular type */ };
__host__ __device__
void ordinary_function(type* tc_operand, int block, int thread) {
    assert(block == 0 && thread == 0);
    // use tc_operand
}
__global__
void entry_point_function(type* tc_operand) {
    ordinary_function(tc_operand, blockIdx.x, threadIdx.x);
}

#include <memory>
template <class T> struct managed {
    typedef T value_type;
    managed () = default;
    template <class U> constexpr managed (const managed<U>&) noexcept {}
    T* allocate(std::size_t n) {
        void* out = nullptr;
        cudaMallocManaged(&out, n*sizeof(T));
        return static_cast<T*>(out);  
    }
    void deallocate(T* p, std::size_t) noexcept { cudaFree(p); }
};
int main() {
    auto managed_object = managed<type>().allocate(1);
    entry_point_function<<<1,1>>>(managed_object);
    cudaDeviceSynchronize();
    ordinary_function(managed_object, 0, 0);
    managed<type>().deallocate(managed_object, 1);
    return 0;
}
