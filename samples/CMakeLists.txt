cmake_minimum_required(VERSION 3.18)

project(libcudacxx-examples LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

set(GPU_ARCHS "70;72;75;80" CACHE STRING "List of GPU architectures to compile for.")

message(STATUS "Enabled CUDA architectures:${GPU_ARCHS}")

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP)

function(ConfigureHostBench BENCH_NAME BENCH_SRC)
   add_executable("${BENCH_NAME}" "${BENCH_SRC}")
   target_link_libraries("${BENCH_NAME}" PRIVATE Threads::Threads)

   # TODO: Link against libcudaxx interface target instead
   target_include_directories("${BENCH_NAME}" PRIVATE
                                              "${CMAKE_CURRENT_SOURCE_DIR}/../include")
   if(OpenMP_CXX_FOUND)
      target_link_libraries("${BENCH_NAME}" PRIVATE OpenMP::OpenMP_CXX)
   endif()
endfunction(ConfigureHostBench)

function(ConfigureDeviceBench BENCH_NAME BENCH_SRC)
   add_executable("${BENCH_NAME}" "${BENCH_SRC}")
   set_property(TARGET "${BENCH_NAME}" PROPERTY CUDA_ARCHITECTURES "${GPU_ARCHS}")
   # TODO: Link against libcudaxx interface target instead
   target_include_directories("${BENCH_NAME}" PRIVATE
                                              "${CMAKE_CURRENT_SOURCE_DIR}/../include")
   target_compile_options("${BENCH_NAME}" PRIVATE --expt-extended-lambda --expt-relaxed-constexpr)
endfunction(ConfigureDeviceBench)


add_executable(trie_st trie_st.cpp)
target_compile_features(trie_st PRIVATE cxx_std_11)
execute_process(COMMAND mkdir -p books)
file(DOWNLOAD https://www.gutenberg.org/files/2600/2600-0.txt books/2600-0.txt)
file(DOWNLOAD http://www.gutenberg.org/cache/epub/996/pg996.txt books/pg996.txt)
file(DOWNLOAD http://www.gutenberg.org/cache/epub/55/pg55.txt books/pg55.txt)
file(DOWNLOAD https://www.gutenberg.org/files/8800/8800.txt books/8800.txt)
file(DOWNLOAD https://www.gutenberg.org/files/84/84-0.txt books/84-0.txt)
file(DOWNLOAD http://www.gutenberg.org/cache/epub/6130/pg6130.txt books/pg6130.txt)
file(DOWNLOAD http://www.gutenberg.org/cache/epub/1727/pg1727.txt books/pg1727.txt)
file(DOWNLOAD https://www.gutenberg.org/files/2701/2701-0.txt books/2701-0.txt)
file(DOWNLOAD https://www.gutenberg.org/files/35/35-0.txt books/35-0.txt)
file(DOWNLOAD https://www.gutenberg.org/files/1342/1342-0.txt books/1342-0.txt)


