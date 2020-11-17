//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <vector>
#include <string>
#include <map>

using namespace std::string_literals;

int main() {

    std::map<std::string, std::string> scopes{ {"system", ".sys"},
                                               {"device", ".gpu"},
                                               {"block", ".cta"} };

    std::map<std::string, std::string> membar_scopes{ {"system", ".sys"},
                                                      {"device", ".gl"},
                                                      {"block", ".cta"} };

    std::map<std::string, std::string> fence_semantics{ { "sc", ".sc" },
                                                        { "acq_rel", ".acq_rel" } };

    bool const ld_as_atom = false;

    std::vector<int> ld_sizes{  //8,
                                //16,
                                32,
                                64
                                };
    std::map<std::string, std::string> ld_semantics{ { "relaxed", ".relaxed" },
                                                     { "acquire", ".acquire" },
                                                     { "volatile", ".volatile" } };

    std::vector<int> st_sizes{  //8,
                                //16,
                                32,
                                64 };
    std::map<std::string, std::string> st_semantics{ { "relaxed", ".relaxed" },
                                                     { "release", ".release" },
                                                     { "volatile", ".volatile" } };

    std::vector<int> rmw_sizes{ 32, 64 };
    std::map<std::string, std::string> rmw_semantics{ { "relaxed", ".relaxed" },
                                                      { "acquire", ".acquire" },
                                                      { "release", ".release" },
                                                      { "acq_rel", ".acq_rel" },
                                                      { "volatile", "" } };
    std::map<std::string, std::string> rmw_operations{ { "exchange", ".exch" },
                                                       { "compare_exchange", ".cas" },
                                                       { "fetch_add", ".add" },
                                                       { "fetch_sub", ".add" },
                                                       { "fetch_max", ".max" },
                                                       { "fetch_min", ".min" },
                                                       { "fetch_and", ".and" },
                                                       { "fetch_or", ".or" },
                                                       { "fetch_xor", ".xor" } };

    std::vector<std::string> cv_qualifier{ "volatile "/*, ""*/ };

    std::map<int, std::string> registers{ //{ 8, "r" } ,
                                          //{ 16, "h" },
                                          { 32, "r" },
                                          { 64, "l" } };

    std::ofstream out("__atomic_generated");

    out << R"XXX(//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
)XXX" << "\n\n";

    auto scopenametag = [&](auto scope) {
        return "__thread_scope_" + scope + "_tag";
    };
    auto fencename = [&](auto sem, auto scope) {
        return "__cuda_fence_" + sem + "_" + scope;
    };

    out << "_LIBCUDACXX_BEGIN_NAMESPACE_CUDA\n";
    out << "namespace detail {\n";
    out << "\n";

    for(auto& s : scopes) {
        out << "static inline __device__ void __cuda_membar_" << s.first << "() { asm volatile(\"membar" << membar_scopes[s.first] << ";\":::\"memory\"); }\n";
        for(auto& sem : fence_semantics)
            out << "static inline __device__ void " << fencename(sem.first, s.first) << "() { asm volatile(\"fence" << sem.second << s.second << ";\":::\"memory\"); }\n";
        out << "static inline __device__ void __atomic_thread_fence_cuda(int __memorder, " << scopenametag(s.first) << ") {\n";
        out << "    switch (__memorder) {\n";
        out << "#if __CUDA_ARCH__ >= 700\n";
        out << "    case __ATOMIC_SEQ_CST: " << fencename("sc"s, s.first) << "(); break;\n";
        out << "    case __ATOMIC_CONSUME:\n";
        out << "    case __ATOMIC_ACQUIRE:\n";
        out << "    case __ATOMIC_ACQ_REL:\n";
        out << "    case __ATOMIC_RELEASE: " << fencename("acq_rel"s, s.first) << "(); break;\n";
        out << "#else\n";
        out << "    case __ATOMIC_SEQ_CST:\n";
        out << "    case __ATOMIC_CONSUME:\n";
        out << "    case __ATOMIC_ACQUIRE:\n";
        out << "    case __ATOMIC_ACQ_REL:\n";
        out << "    case __ATOMIC_RELEASE: __cuda_membar_" << s.first << "(); break;\n";
        out << "#endif // __CUDA_ARCH__ >= 700\n";
        out << "    case __ATOMIC_RELAXED: break;\n";
        out << "    default: assert(0);\n";
        out << "    }\n";
        out << "}\n";
        for(auto& sz : ld_sizes) {
            for(auto& sem : ld_semantics) {
                out << "template<class _CUDA_A, class _CUDA_B> ";
                out << "static inline __device__ void __cuda_load_" << sem.first << "_" << sz << "_" << s.first << "(_CUDA_A __ptr, _CUDA_B& __dst) {";
                if(ld_as_atom)
                    out << "asm volatile(\"atom.add" << (sem.first == "volatile" ? "" : sem.second.c_str()) << s.second << ".u" << sz << " %0, [%1], 0;\" : ";
                else
                    out << "asm volatile(\"ld" << sem.second << (sem.first == "volatile" ? "" : s.second.c_str()) << ".b" << sz << " %0,[%1];\" : ";
                out << "\"=" << registers[sz] << "\"(__dst) : \"l\"(__ptr)";
                out << " : \"memory\"); }\n";
            }
            for(auto& cv: cv_qualifier) {
                out << "template<class _Type, typename _CUDA_VSTD::enable_if<sizeof(_Type)==" << sz/8 << ", int>::type = 0>\n";
                out << "__device__ void __atomic_load_cuda(const " << cv << "_Type *__ptr, _Type *__ret, int __memorder, " << scopenametag(s.first) << ") {\n";
                out << "    uint" << (registers[sz] == "r" ? 32 : sz) << "_t __tmp = 0;\n";
                out << "    switch (__memorder) {\n";
                out << "#if __CUDA_ARCH__ >= 700\n";
                out << "    case __ATOMIC_SEQ_CST: " << fencename("sc"s, s.first) << "();\n";
                out << "    case __ATOMIC_CONSUME:\n";
                out << "    case __ATOMIC_ACQUIRE: __cuda_load_acquire_" << sz << "_" << s.first << "(__ptr, __tmp); break;\n";
                out << "    case __ATOMIC_RELAXED: __cuda_load_relaxed_" << sz << "_" << s.first << "(__ptr, __tmp); break;\n";
                out << "#else\n";
                out << "    case __ATOMIC_SEQ_CST: __cuda_membar_" << s.first << "();\n";
                out << "    case __ATOMIC_CONSUME:\n";
                out << "    case __ATOMIC_ACQUIRE: __cuda_load_volatile_" << sz << "_" << s.first << "(__ptr, __tmp); __cuda_membar_" << s.first << "(); break;\n";
                out << "    case __ATOMIC_RELAXED: __cuda_load_volatile_" << sz << "_" << s.first << "(__ptr, __tmp); break;\n";
                out << "#endif // __CUDA_ARCH__ >= 700\n";
                out << "    default: assert(0);\n";
                out << "    }\n";
                out << "    memcpy(__ret, &__tmp, " << sz/8 << ");\n";
                out << "}\n";
            }
        }
        for(auto& sz : st_sizes) {
            for(auto& sem : st_semantics) {
                out << "template<class _CUDA_A, class _CUDA_B> ";
                out << "static inline __device__ void __cuda_store_" << sem.first << "_" << sz << "_" << s.first << "(_CUDA_A __ptr, _CUDA_B __src) { ";
                out << "asm volatile(\"st" << sem.second << (sem.first == "volatile" ? "" : s.second.c_str()) << ".b" << sz << " [%0], %1;\" :: ";
                out << "\"l\"(__ptr),\"" << registers[sz] << "\"(__src)";
                out << " : \"memory\"); }\n";
            }
            for(auto& cv: cv_qualifier) {
                out << "template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==" << sz/8 << ", int>::type = 0>\n";
                out << "__device__ void __atomic_store_cuda(" << cv << "_Type *__ptr, _Type *__val, int __memorder, " << scopenametag(s.first) << ") {\n";
                out << "    uint" << (registers[sz] == "r" ? 32 : sz) << "_t __tmp = 0;\n";
                out << "    memcpy(&__tmp, __val, " << sz/8 << ");\n";
                out << "    switch (__memorder) {\n";
                out << "#if __CUDA_ARCH__ >= 700\n";
                out << "    case __ATOMIC_RELEASE: __cuda_store_release_" << sz << "_" << s.first << "(__ptr, __tmp); break;\n";
                out << "    case __ATOMIC_SEQ_CST: " << fencename("sc"s, s.first) << "();\n";
                out << "    case __ATOMIC_RELAXED: __cuda_store_relaxed_" << sz << "_" << s.first << "(__ptr, __tmp); break;\n";
                out << "#else\n";
                out << "    case __ATOMIC_RELEASE:\n";
                out << "    case __ATOMIC_SEQ_CST: __cuda_membar_" << s.first << "();\n";
                out << "    case __ATOMIC_RELAXED: __cuda_store_volatile_" << sz << "_" << s.first << "(__ptr, __tmp); break;\n";
                out << "#endif // __CUDA_ARCH__ >= 700\n";
                out << "    default: assert(0);\n";
                out << "    }\n";
                out << "}\n";
            }
        }
        for(auto& sz : rmw_sizes) {
            for(auto& rmw: rmw_operations) {
                for(auto& sem : rmw_semantics) {
                    if(rmw.first == "compare_exchange")
                        out << "template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> ";
                    else
                        out << "template<class _CUDA_A, class _CUDA_B, class _CUDA_C> ";
                    out << "static inline __device__ void __cuda_" << rmw.first << "_" << sem.first << "_" << sz << "_" << s.first << "(";
                    if(rmw.first == "compare_exchange")
                        out << "_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op";
                    else
                        out << "_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op";
                    out << ") { ";
                    if(rmw.first == "fetch_sub")
                        out << "__op = -__op;" << std::endl;
                    if(rmw.first == "fetch_add" || rmw.first == "fetch_sub" || rmw.first == "fetch_max" || rmw.first == "fetch_min")
                        out << "asm volatile(\"atom" << rmw.second << sem.second << s.second << ".u" << sz << " ";
                    else
                        out << "asm volatile(\"atom" << rmw.second << sem.second << s.second << ".b" << sz << " ";
                    if(rmw.first == "compare_exchange")
                        out << "%0,[%1],%2,%3";
                    else
                        out << "%0,[%1],%2";
                    out << ";\" : ";
                    if(rmw.first == "compare_exchange")
                        out << "\"=" << registers[sz] << "\"(__dst) : \"l\"(__ptr),\"" << registers[sz] << "\"(__cmp),\"" << registers[sz] << "\"(__op)";
                    else
                        out << "\"=" << registers[sz] << "\"(__dst) : \"l\"(__ptr),\"" << registers[sz] << "\"(__op)";
                    out << " : \"memory\"); }\n";
                }
                for(auto& cv: cv_qualifier) {
                    if(rmw.first == "compare_exchange") {
                        out << "template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==" << sz/8 << ", int>::type = 0>\n";
                        out << "__device__ bool __atomic_compare_exchange_cuda(" << cv << "_Type *__ptr, _Type *__expected, const _Type *__desired, bool, int __success_memorder, int __failure_memorder, " << scopenametag(s.first) << ") {\n";
                        out << "    uint" << sz << "_t __tmp = 0, __old = 0, __old_tmp;\n";
                        out << "    memcpy(&__tmp, __desired, " << sz/8 << ");\n";
                        out << "    memcpy(&__old, __expected, " << sz/8 << ");\n";
                        out << "    __old_tmp = __old;\n";
                        out << "    switch (__stronger_order_cuda(__success_memorder, __failure_memorder)) {\n";
                        out << "#if __CUDA_ARCH__ >= 700\n";
                        out << "    case __ATOMIC_SEQ_CST: " << fencename("sc"s, s.first) << "();\n";
                        out << "    case __ATOMIC_CONSUME:\n";
                        out << "    case __ATOMIC_ACQUIRE: __cuda_compare_exchange_acquire_" << sz << "_" << s.first << "(__ptr, __old, __old_tmp, __tmp); break;\n";
                        out << "    case __ATOMIC_ACQ_REL: __cuda_compare_exchange_acq_rel_" << sz << "_" << s.first << "(__ptr, __old, __old_tmp, __tmp); break;\n";
                        out << "    case __ATOMIC_RELEASE: __cuda_compare_exchange_release_" << sz << "_" << s.first << "(__ptr, __old, __old_tmp, __tmp); break;\n";
                        out << "    case __ATOMIC_RELAXED: __cuda_compare_exchange_relaxed_" << sz << "_" << s.first << "(__ptr, __old, __old_tmp, __tmp); break;\n";
                        out << "#else\n";
                        out << "    case __ATOMIC_SEQ_CST:\n";
                        out << "    case __ATOMIC_ACQ_REL: __cuda_membar_" << s.first << "();\n";
                        out << "    case __ATOMIC_CONSUME:\n";
                        out << "    case __ATOMIC_ACQUIRE: __cuda_compare_exchange_volatile_" << sz << "_" << s.first << "(__ptr, __old, __old_tmp, __tmp); __cuda_membar_" << s.first << "(); break;\n";
                        out << "    case __ATOMIC_RELEASE: __cuda_membar_" << s.first << "(); __cuda_compare_exchange_volatile_" << sz << "_" << s.first << "(__ptr, __old, __old_tmp, __tmp); break;\n";
                        out << "    case __ATOMIC_RELAXED: __cuda_compare_exchange_volatile_" << sz << "_" << s.first << "(__ptr, __old, __old_tmp, __tmp); break;\n";
                        out << "#endif // __CUDA_ARCH__ >= 700\n";
                        out << "    default: assert(0);\n";
                        out << "    }\n";
                        out << "    bool const __ret = __old == __old_tmp;\n";
                        out << "    memcpy(__expected, &__old, " << sz/8 << ");\n";
                        out << "    return __ret;\n";
                        out << "}\n";
                    }
                    else {
                        out << "template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==" << sz/8 << ", int>::type = 0>\n";
                        if(rmw.first == "exchange") {
                            out << "__device__ void __atomic_exchange_cuda(" << cv << "_Type *__ptr, _Type *__val, _Type *__ret, int __memorder, " << scopenametag(s.first) << ") {\n";
                            out << "    uint" << sz << "_t __tmp = 0;\n";
                            out << "    memcpy(&__tmp, __val, " << sz/8 << ");\n";
                        }
                        else {
                            out << "__device__ _Type __atomic_" << rmw.first << "_cuda(" << cv << "_Type *__ptr, _Type __val, int __memorder, " << scopenametag(s.first) << ") {\n";
                            out << "    _Type __ret;\n";
                            out << "    uint" << sz << "_t __tmp = 0;\n";
                            out << "    memcpy(&__tmp, &__val, " << sz/8 << ");\n";
                        }
                        out << "    switch (__memorder) {\n";
                        out << "#if __CUDA_ARCH__ >= 700\n";
                        out << "    case __ATOMIC_SEQ_CST: " << fencename("sc"s, s.first) << "();\n";
                        out << "    case __ATOMIC_CONSUME:\n";
                        out << "    case __ATOMIC_ACQUIRE: __cuda_" << rmw.first << "_acquire_" << sz << "_" << s.first << "(__ptr, __tmp, __tmp); break;\n";
                        out << "    case __ATOMIC_ACQ_REL: __cuda_" << rmw.first << "_acq_rel_" << sz << "_" << s.first << "(__ptr, __tmp, __tmp); break;\n";
                        out << "    case __ATOMIC_RELEASE: __cuda_" << rmw.first << "_release_" << sz << "_" << s.first << "(__ptr, __tmp, __tmp); break;\n";
                        out << "    case __ATOMIC_RELAXED: __cuda_" << rmw.first << "_relaxed_" << sz << "_" << s.first << "(__ptr, __tmp, __tmp); break;\n";
                        out << "#else\n";
                        out << "    case __ATOMIC_SEQ_CST:\n";
                        out << "    case __ATOMIC_ACQ_REL: __cuda_membar_" << s.first << "();\n";
                        out << "    case __ATOMIC_CONSUME:\n";
                        out << "    case __ATOMIC_ACQUIRE: __cuda_" << rmw.first << "_volatile_" << sz << "_" << s.first << "(__ptr, __tmp, __tmp); __cuda_membar_" << s.first << "(); break;\n";
                        out << "    case __ATOMIC_RELEASE: __cuda_membar_" << s.first << "(); __cuda_" << rmw.first << "_volatile_" << sz << "_" << s.first << "(__ptr, __tmp, __tmp); break;\n";
                        out << "    case __ATOMIC_RELAXED: __cuda_" << rmw.first << "_volatile_" << sz << "_" << s.first << "(__ptr, __tmp, __tmp); break;\n";
                        out << "#endif // __CUDA_ARCH__ >= 700\n";
                        out << "    default: assert(0);\n";
                        out << "    }\n";
                        if(rmw.first == "exchange")
                            out << "    memcpy(__ret, &__tmp, " << sz/8 << ");\n";
                        else {
                            out << "    memcpy(&__ret, &__tmp, " << sz/8 << ");\n";
                            out << "    return __ret;\n";
                        }
                        out << "}\n";
                    }
                }
            }
        }
        for(auto& cv: cv_qualifier) {
            std::vector<std::string> addsub{ "add", "sub" };
            for(auto& op : addsub) {
                out << "template<class _Type>\n";
                out << "__device__ _Type* __atomic_fetch_" << op << "_cuda(_Type *" << cv << "*__ptr, ptrdiff_t __val, int __memorder, " << scopenametag(s.first) << ") {\n";
                out << "    _Type* __ret;\n";
                out << "    uint64_t __tmp = 0;\n";
                out << "    memcpy(&__tmp, &__val, 8);\n";
                if(op == "sub")
                    out << "    __tmp = -__tmp;\n";
                out << "    __tmp *= sizeof(_Type);\n";
                out << "    switch (__memorder) {\n";
                out << "#if __CUDA_ARCH__ >= 700\n";
                out << "    case __ATOMIC_SEQ_CST: " << fencename("sc"s, s.first) << "();\n";
                out << "    case __ATOMIC_CONSUME:\n";
                out << "    case __ATOMIC_ACQUIRE: __cuda_fetch_add_acquire_64_" << s.first << "(__ptr, __tmp, __tmp); break;\n";
                out << "    case __ATOMIC_ACQ_REL: __cuda_fetch_add_acq_rel_64_" << s.first << "(__ptr, __tmp, __tmp); break;\n";
                out << "    case __ATOMIC_RELEASE: __cuda_fetch_add_release_64_" << s.first << "(__ptr, __tmp, __tmp); break;\n";
                out << "    case __ATOMIC_RELAXED: __cuda_fetch_add_relaxed_64_" << s.first << "(__ptr, __tmp, __tmp); break;\n";
                out << "#else\n";
                out << "    case __ATOMIC_SEQ_CST:\n";
                out << "    case __ATOMIC_ACQ_REL: __cuda_membar_" << s.first << "();\n";
                out << "    case __ATOMIC_CONSUME:\n";
                out << "    case __ATOMIC_ACQUIRE: __cuda_fetch_add_volatile_64_" << s.first << "(__ptr, __tmp, __tmp); __cuda_membar_" << s.first << "(); break;\n";
                out << "    case __ATOMIC_RELEASE: __cuda_membar_" << s.first << "(); __cuda_fetch_add_volatile_64_" << s.first << "(__ptr, __tmp, __tmp); break;\n";
                out << "    case __ATOMIC_RELAXED: __cuda_fetch_add_volatile_64_" << s.first << "(__ptr, __tmp, __tmp); break;\n";
                out << "#endif // __CUDA_ARCH__ >= 700\n";
                out << "    default: assert(0);\n";
                out << "    }\n";
                out << "    memcpy(&__ret, &__tmp, 8);\n";
                out << "    return __ret;\n";
                out << "}\n";
            }
        }
    }

    out << "\n";
    out << "}\n";
    out << "_LIBCUDACXX_END_NAMESPACE_CUDA\n";

    return 0;
}
