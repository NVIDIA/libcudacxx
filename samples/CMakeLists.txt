cmake_minimum_required(VERSION 3.8)
project(libcudacxx-samples CXX)

if (NOT "${CMAKE_CUDA_HOST_COMPILER}" STREQUAL "")
  unset(CMAKE_CUDA_HOST_COMPILER CACHE)
  message(FATAL_ERROR "Thrust tests and examples require the C++ compiler"
    " and the CUDA host compiler to be the same; to set this compiler, please"
    " use the CMAKE_CXX_COMPILER variable, not the CMAKE_CUDA_HOST_COMPILER"
    " variable.")
  endif ()
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
enable_language(CUDA)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")

set(HIGHEST_COMPUTE_ARCH 80)
set(KNOWN_COMPUTE_ARCHS 70 72 75 80)

option(DISABLE_ARCH_BY_DEFAULT "If ON, then all CUDA architectures are disabled on the initial CMake run." OFF)
set(OPTION_INIT ON)
if (DISABLE_ARCH_BY_DEFAULT)
  set(OPTION_INIT OFF)
endif ()
if (NOT ${HIGHEST_COMPUTE_ARCH} IN_LIST KNOWN_COMPUTE_ARCHS)
  message(FATAL_ERROR "When changing the highest compute version, don't forget to add it to the list!")
endif ()

foreach (COMPUTE_ARCH IN LISTS KNOWN_COMPUTE_ARCHS)
  option(ENABLE_COMPUTE_${COMPUTE_ARCH} "Enable code generation for tests for sm_${COMPUTE_ARCH}" ${OPTION_INIT})
  if (ENABLE_COMPUTE_${COMPUTE_ARCH})
    set(COMPUTE_ARCHS "${COMPUTE_ARCHS} ${COMPUTE_ARCH}")
    set(COMPUTE_MESSAGE "${COMPUTE_MESSAGE} sm_${COMPUTE_ARCH}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${COMPUTE_ARCH},code=sm_${COMPUTE_ARCH}")
  endif ()
endforeach ()

option(ENABLE_COMPUTE_FUTURE "Enable code generation for tests for compute_${HIGHEST_COMPUTE_ARCH}" ${OPTION_INIT})
if (ENABLE_COMPUTE_FUTURE)
  set(COMPUTE_MESSAGE "${COMPUTE_MESSAGE} compute_${HIGHEST_COMPUTE_ARCH}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${HIGHEST_COMPUTE_ARCH},code=compute_${HIGHEST_COMPUTE_ARCH}")
endif ()

message(STATUS "Enabled CUDA architectures:${COMPUTE_MESSAGE}")

include_directories(${CMAKE_SOURCE_DIR}/../include)

find_package(Threads REQUIRED)
find_package(OpenMP)

add_executable(benchmark_host benchmark.cpp)

target_link_libraries(benchmark_host PUBLIC Threads::Threads)
if(OpenMP_CXX_FOUND)
    target_link_libraries(benchmark_host PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(benchmark_device benchmark.cu)
