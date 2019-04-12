cmake_minimum_required(VERSION 3.8)
project(libcuxx C CXX)

set(PACKAGE_NAME libcuxx)
set(PACKAGE_VERSION 10.3)
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")

set(LIBCXX_HIGHEST_COMPUTE_ARCH 75)
set(LIBCXX_KNOWN_COMPUTE_ARCHS 30 32 35 50 52 53 60 61 62 70 72 75)

option(LIBCXX_DISABLE_ARCH_BY_DEFAULT "If ON, then all CUDA architectures are disabled on the initial CMake run." OFF)
set(OPTION_INIT ON)
if (LIBCXX_DISABLE_ARCH_BY_DEFAULT)
  set(OPTION_INIT OFF)
endif ()

if (NOT ${LIBCXX_HIGHEST_COMPUTE_ARCH} IN_LIST LIBCXX_KNOWN_COMPUTE_ARCHS)
  message(FATAL_ERROR "When changing the highest compute version, don't forget to add it to the list!")
endif ()

foreach (COMPUTE_ARCH IN LISTS LIBCXX_KNOWN_COMPUTE_ARCHS)
  option(LIBCXX_ENABLE_COMPUTE_${COMPUTE_ARCH} "Enable code generation for tests for sm_${COMPUTE_ARCH}" ${OPTION_INIT})
  if (LIBCXX_ENABLE_COMPUTE_${COMPUTE_ARCH})
    set(LIBCXX_COMPUTE_ARCH_FLAGS "${LIBCXX_COMPUTE_ARCH_FLAGS} -gencode arch=compute_${COMPUTE_ARCH},code=sm_${COMPUTE_ARCH}")
    set(COMPUTE_MESSAGE "${COMPUTE_MESSAGE} sm_${COMPUTE_ARCH}")
  endif ()
endforeach ()

option(LIBCXX_ENABLE_COMPUTE_FUTURE "Enable code generation for tests for compute_${LIBCXX_HIGHEST_COMPUTE_ARCH}" ${OPTION_INIT})
if (LIBCXX_ENABLE_COMPUTE_FUTURE)
  set(LIBCXX_COMPUTE_ARCH_FLAGS
    "${LIBCXX_COMPUTE_ARCH_FLAGS} -gencode arch=compute_${LIBCXX_HIGHEST_COMPUTE_ARCH},code=compute_${LIBCXX_HIGHEST_COMPUTE_ARCH}")
  set(COMPUTE_MESSAGE "${COMPUTE_MESSAGE} compute_${LIBCXX_HIGHEST_COMPUTE_ARCH}")
endif ()

message(STATUS "-- Enabled CUDA architectures:${COMPUTE_MESSAGE}")

# TODO: this has to be reconfigurable with other options
# this has to land in cache, but has to be forced, and controlled by other variables
set(LIBCXX_TEST_COMPILER_FLAGS
    "-ccbin ${LIBCXX_HOST_COMPILER} ${LIBCXX_COMPUTE_ARCH_FLAGS} -include ${CMAKE_SOURCE_DIR}/test/force_include.h -I${CMAKE_SOURCE_DIR}/include"
    CACHE STRING "Flags for libcxx testing.")

set(LIBCXX_TEST_LINKER_FLAGS
    "-ccbin ${LIBCXX_HOST_COMPILER} -Xcompiler \\\\\\\"-Wl,-rpath,${LIBCXX_TEST_LD_LIBRARY_PATH}\\\\\\\""
    CACHE STRING "Flags for libcxx testing.")

set(LIBCXX_ENABLE_EXPERIMENTAL_LIBRARY OFF CACHE BOOL "")
set(LIBCXX_ENABLE_EXCEPTIONS OFF CACHE BOOL "")
set(LIBCXX_ENABLE_FILESYSTEM OFF CACHE BOOL "")
set(LIBCXX_ENABLE_RTTI OFF CACHE BOOL "")
set(LIBCXX_ENABLE_SHARED OFF CACHE BOOL "")
set(LIBCXX_HAS_ATOMIC_LIB FALSE CACHE BOOL "")

set(LIBCXX_SOURCE_DIR ${CMAKE_SOURCE_DIR} CACHE STRING "")

add_subdirectory(libcxx)
