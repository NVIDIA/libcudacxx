cmake_minimum_required(VERSION 3.8)
project(libcuxx C CXX)

set(PACKAGE_NAME libcuxx)
set(PACKAGE_VERSION 10.3)
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")

# TODO: this has to be reconfigurable with other options
# this has to land in cache, but has to be forced, and controlled by other variables
set(LIBCXX_TEST_COMPILER_FLAGS
    "-include ${CMAKE_SOURCE_DIR}/test/force_include.h -I${CMAKE_SOURCE_DIR}/include -gencode arch=compute_70,code=sm_70"
    CACHE STRING "Additional forced flags for libcxx testing.")
set(LIBCXX_ENABLE_EXPERIMENTAL_LIBRARY OFF CACHE BOOL "")
set(LIBCXX_ENABLE_EXCEPTIONS OFF CACHE BOOL "")
set(LIBCXX_ENABLE_FILESYSTEM OFF CACHE BOOL "")
set(LIBCXX_ENABLE_RTTI OFF CACHE BOOL "")
set(LIBCXX_ENABLE_SHARED OFF CACHE BOOL "")
set(LIBCXX_HAS_ATOMIC_LIB FALSE CACHE BOOL "")

set(LIBCXX_SOURCE_DIR ${CMAKE_SOURCE_DIR} CACHE STRING "")

add_subdirectory(libcxx)
