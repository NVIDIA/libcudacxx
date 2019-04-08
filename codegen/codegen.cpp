/*

Copyright (c) 2018-2019, NVIDIA Corporation

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

#include <fstream>
#include <vector>
#include <string>
#include <map>

using namespace std::string_literals;

int main() {

    std::map<std::string, std::string> scopes{ {"system", "sys"},
                                               {"device", "gpu"},
                                               {"block", "cta"} };

    std::vector<std::string> fence_semantics{ "sc", "acq_rel" };

    std::vector<int> ld_sizes{ 8, 16, 32, 64 };
    std::vector<std::string> ld_semantics{ "relaxed", "acquire" };

    std::vector<int> st_sizes{ 8, 16, 32, 64 };
    std::vector<std::string> st_semantics{ "relaxed", "release" };

    std::vector<int> rmw_sizes{ 32, 64 };
    std::vector<std::string> rmw_semantics{ "relaxed", "acquire", "release", "acq_rel" };
    std::map<std::string, std::string> rmw_operations{ { "exchange", "exch" },
                                                       { "compare_exchange", "cas" },
                                                       { "fetch_add", "add" },
                                                       { "fetch_sub", "add" },
                                                       { "fetch_and", "and" },
                                                       { "fetch_or", "or" },
                                                       { "fetch_xor", "xor" } };

    std::vector<std::string> cv_qualifier{ "volatile "/*, ""*/ };

    std::map<int, std::string> registers{ { 8, "r" } ,
                                          { 16, "h" },
                                          { 32, "r" },
                                          { 64, "l" } };

    std::ofstream out("__atomic_generated");

    out << R"XXX(/*

Copyright (c) 2018-2019, NVIDIA Corporation

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

*/)XXX" << "\n\n";

    auto scopenametag = [&](auto scope) {
        return "__thread_scope_" + scope + "_tag";
    };
    auto fencename = [&](auto sem, auto scope) {
        return "__cuda_fence_" + sem + "_" + scope;
    };

    out << "_LIBCPP_BEGIN_NAMESPACE_CUDA\n";
    out << "namespace detail {\n";
    out << "\n";

    for(auto& s : scopes) {
        for(auto& sem : fence_semantics)
            out << "static inline __device__ void " << fencename(sem, s.first) << "() { asm volatile(\"fence." << sem << "." << s.second << ";\":::\"memory\"); }\n";
        out << "static inline __device__ void __atomic_thread_fence_cuda(int memorder, " << scopenametag(s.first) << ") {\n";
        out << "    switch (memorder) {\n";
        out << "    case __ATOMIC_SEQ_CST: " << fencename("sc"s, s.first) << "(); break;\n";
        out << "    case __ATOMIC_CONSUME:\n";
        out << "    case __ATOMIC_ACQUIRE:\n";
        out << "    case __ATOMIC_ACQ_REL:\n";
        out << "    case __ATOMIC_RELEASE: " << fencename("acq_rel"s, s.first) << "(); break;\n";
        out << "    case __ATOMIC_RELAXED: break;\n";
        out << "    default: assert(0);\n";
        out << "    }\n";
        out << "}\n";
        for(auto& sz : ld_sizes) {
            for(auto& sem : ld_semantics) {
                out << "template<class _CUDA_A, class _CUDA_B> ";
                out << "static inline __device__ void __cuda_load_" << sem << "_" << sz << "_" << s.first << "(_CUDA_A _ptr, _CUDA_B& _dst) {";
                out << "asm volatile(\"ld." << sem << "." << s.second << ".b" << sz << " %0,[%1];\" : ";
                out << "\"=" << registers[sz] << "\"(_dst) : \"l\"(_ptr)";
                out << " : \"memory\"); }\n";
            }
            for(auto& cv: cv_qualifier) {
                out << "template<class type, typename _VSTD::enable_if<sizeof(type)==" << sz/8 << ", int>::type = 0>\n";
                out << "__device__ void __atomic_load_cuda(const " << cv << "type *ptr, type *ret, int memorder, " << scopenametag(s.first) << ") {\n";
                out << "    uint" << (registers[sz] == "r" ? 32 : sz) << "_t tmp = 0;\n";
                out << "    switch (memorder) {\n";
                out << "    case __ATOMIC_SEQ_CST: " << fencename("sc"s, s.first) << "();\n";
                out << "    case __ATOMIC_CONSUME:\n";
                out << "    case __ATOMIC_ACQUIRE: __cuda_load_acquire_" << sz << "_" << s.first << "(ptr, tmp); break;\n";
                out << "    case __ATOMIC_RELAXED: __cuda_load_relaxed_" << sz << "_" << s.first << "(ptr, tmp); break;\n";
                out << "    default: assert(0);\n";
                out << "    }\n";
                out << "    memcpy(ret, &tmp, " << sz/8 << ");\n";
                out << "}\n";
            }
        }
        for(auto& sz : st_sizes) {
            for(auto& sem : st_semantics) {
                out << "template<class _CUDA_A, class _CUDA_B> ";
                out << "static inline __device__ void __cuda_store_" << sem << "_" << sz << "_" << s.first << "(_CUDA_A _ptr, _CUDA_B _src) { ";
                out << "asm volatile(\"st." << sem << "." << s.second << ".b" << sz << " [%0], %1;\" :: ";
                out << "\"l\"(_ptr),\"" << registers[sz] << "\"(_src)";
                out << " : \"memory\"); }\n";
            }
            for(auto& cv: cv_qualifier) {
                out << "template<class type, typename cuda::std::enable_if<sizeof(type)==" << sz/8 << ", int>::type = 0>\n";
                out << "__device__ void __atomic_store_cuda(" << cv << "type *ptr, type *val, int memorder, " << scopenametag(s.first) << ") {\n";
                out << "    uint" << (registers[sz] == "r" ? 32 : sz) << "_t tmp = 0;\n";
                out << "    memcpy(&tmp, val, " << sz/8 << ");\n";
                out << "    switch (memorder) {\n";
                out << "    case __ATOMIC_RELEASE: __cuda_store_release_" << sz << "_" << s.first << "(ptr, tmp); break;\n";
                out << "    case __ATOMIC_SEQ_CST: " << fencename("sc"s, s.first) << "();\n";
                out << "    case __ATOMIC_RELAXED: __cuda_store_relaxed_" << sz << "_" << s.first << "(ptr, tmp); break;\n";
                out << "    default: assert(0);\n";
                out << "    }\n";
                out << "}\n";
            }
        }
        for(auto& sz : rmw_sizes) {
            for(auto& rmw: rmw_operations) {
                if(rmw.first != "fetch_sub")
                    for(auto& sem : rmw_semantics) {
                        if(rmw.first == "compare_exchange")
                            out << "template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> ";
                        else
                            out << "template<class _CUDA_A, class _CUDA_B, class _CUDA_C> ";
                        out << "static inline __device__ void __cuda_" << rmw.second << "_" << sem << "_" << sz << "_" << s.first << "(";
                        if(rmw.first == "compare_exchange")
                            out << "_CUDA_A _ptr, _CUDA_B& _dst, _CUDA_C _cmp, _CUDA_D _op";
                        else
                            out << "_CUDA_A _ptr, _CUDA_B& _dst, _CUDA_C _op";
                        out << ") { ";
                        if(rmw.first == "fetch_add" || rmw.first == "fetch_sub")
                            out << "asm volatile(\"atom." << rmw.second << "." << sem << "." << s.second << ".u" << sz << " ";
                        else
                            out << "asm volatile(\"atom." << rmw.second << "." << sem << "." << s.second << ".b" << sz << " ";
                        if(rmw.first == "compare_exchange")
                            out << "%0,[%1],%2,%3";
                        else
                            out << "%0,[%1],%2";
                        out << ";\" : ";
                        if(rmw.first == "compare_exchange")
                            out << "\"=" << registers[sz] << "\"(_dst) : \"l\"(_ptr),\"" << registers[sz] << "\"(_cmp),\"" << registers[sz] << "\"(_op)";
                        else
                            out << "\"=" << registers[sz] << "\"(_dst) : \"l\"(_ptr),\"" << registers[sz] << "\"(_op)";
                        out << " : \"memory\"); }\n";
                    }
                for(auto& cv: cv_qualifier) {
                    if(rmw.first == "compare_exchange") {
                        out << "template<class type, typename cuda::std::enable_if<sizeof(type)==" << sz/8 << ", int>::type = 0>\n";
                        out << "__device__ bool __atomic_compare_exchange_cuda(" << cv << "type *ptr, type *expected, const type *desired, bool, int success_memorder, int failure_memorder, " << scopenametag(s.first) << ") {\n";
                        out << "    uint" << sz << "_t tmp = 0, old = 0, old_tmp;\n";
                        out << "    memcpy(&tmp, desired, " << sz/8 << ");\n";
                        out << "    memcpy(&old, expected, " << sz/8 << ");\n";
                        out << "    old_tmp = old;\n";
                        out << "    switch (__stronger_order_cuda(success_memorder, failure_memorder)) {\n";
                        out << "    case __ATOMIC_SEQ_CST: " << fencename("sc"s, s.first) << "();\n";
                        out << "    case __ATOMIC_CONSUME:\n";
                        out << "    case __ATOMIC_ACQUIRE: __cuda_cas_acquire_" << sz << "_" << s.first << "(ptr, old, old_tmp, tmp); break;\n";
                        out << "    case __ATOMIC_ACQ_REL: __cuda_cas_acq_rel_" << sz << "_" << s.first << "(ptr, old, old_tmp, tmp); break;\n";
                        out << "    case __ATOMIC_RELEASE: __cuda_cas_release_" << sz << "_" << s.first << "(ptr, old, old_tmp, tmp); break;\n";
                        out << "    case __ATOMIC_RELAXED: __cuda_cas_relaxed_" << sz << "_" << s.first << "(ptr, old, old_tmp, tmp); break;\n";
                        out << "    default: assert(0);\n";
                        out << "    }\n";
                        out << "    bool const ret = old == old_tmp;\n";
                        out << "    memcpy(expected, &old, " << sz/8 << ");\n";
                        out << "    return ret;\n";
                        out << "}\n";
                    }
                    else {
                        out << "template<class type, typename cuda::std::enable_if<sizeof(type)==" << sz/8 << ", int>::type = 0>\n";
                        if(rmw.first == "exchange") {
                            out << "__device__ void __atomic_exchange_cuda(" << cv << "type *ptr, type *val, type *ret, int memorder, " << scopenametag(s.first) << ") {\n";
                            out << "    uint" << sz << "_t tmp = 0;\n";
                            out << "    memcpy(&tmp, val, " << sz/8 << ");\n";
                        }
                        else {
                            out << "__device__ type __atomic_" << rmw.first << "_cuda(" << cv << "type *ptr, type val, int memorder, " << scopenametag(s.first) << ") {\n";
                            out << "    type ret;\n";
                            out << "    uint" << sz << "_t tmp = 0;\n";
                            out << "    memcpy(&tmp, &val, " << sz/8 << ");\n";
                        }
                        if(rmw.first == "fetch_sub")
                            out << "    tmp = -tmp;\n";
                        out << "    switch (memorder) {\n";
                        out << "    case __ATOMIC_SEQ_CST: " << fencename("sc"s, s.first) << "();\n";
                        out << "    case __ATOMIC_CONSUME:\n";
                        out << "    case __ATOMIC_ACQUIRE: __cuda_" << rmw.second << "_acquire_" << sz << "_" << s.first << "(ptr, tmp, tmp); break;\n";
                        out << "    case __ATOMIC_ACQ_REL: __cuda_" << rmw.second << "_acq_rel_" << sz << "_" << s.first << "(ptr, tmp, tmp); break;\n";
                        out << "    case __ATOMIC_RELEASE: __cuda_" << rmw.second << "_release_" << sz << "_" << s.first << "(ptr, tmp, tmp); break;\n";
                        out << "    case __ATOMIC_RELAXED: __cuda_" << rmw.second << "_relaxed_" << sz << "_" << s.first << "(ptr, tmp, tmp); break;\n";
                        out << "    default: assert(0);\n";
                        out << "    }\n";
                        if(rmw.first == "exchange")
                            out << "    memcpy(ret, &tmp, " << sz/8 << ");\n";
                        else {
                            out << "    memcpy(&ret, &tmp, " << sz/8 << ");\n";
                            out << "    return ret;\n";
                        }
                        out << "}\n";
                    }
                }
            }
        }
        for(auto& cv: cv_qualifier) {
            std::vector<std::string> addsub{ "add", "sub" };
            for(auto& op : addsub) {
                out << "template<class type>\n";
                out << "__device__ type* __atomic_fetch_" << op << "_cuda(type *" << cv << "*ptr, ptrdiff_t val, int memorder, " << scopenametag(s.first) << ") {\n";
                out << "    type* ret;\n";
                out << "    uint64_t tmp = 0;\n";
                out << "    memcpy(&tmp, &val, 8);\n";
                if(op == "sub")
                    out << "    tmp = -tmp;\n";
                out << "    tmp *= sizeof(type);\n";
                out << "    switch (memorder) {\n";
                out << "    case __ATOMIC_SEQ_CST: " << fencename("sc"s, s.first) << "();\n";
                out << "    case __ATOMIC_CONSUME:\n";
                out << "    case __ATOMIC_ACQUIRE: __cuda_add_acquire_64_" << s.first << "(ptr, tmp, tmp); break;\n";
                out << "    case __ATOMIC_ACQ_REL: __cuda_add_acq_rel_64_" << s.first << "(ptr, tmp, tmp); break;\n";
                out << "    case __ATOMIC_RELEASE: __cuda_add_release_64_" << s.first << "(ptr, tmp, tmp); break;\n";
                out << "    case __ATOMIC_RELAXED: __cuda_add_relaxed_64_" << s.first << "(ptr, tmp, tmp); break;\n";
                out << "    default: assert(0);\n";
                out << "    }\n";
                out << "    memcpy(&ret, &tmp, 8);\n";
                out << "    return ret;\n";
                out << "}\n";
            }
        }
    }

    out << "\n";
    out << "}\n";
    out << "_LIBCPP_END_NAMESPACE_CUDA\n";

    return 0;
}
