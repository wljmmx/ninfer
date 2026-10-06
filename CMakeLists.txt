cmake_minimum_required(VERSION 3.28)

# The current implementation supports two single-GPU targets: RTX 4090 class Ada
# devices (sm_89) and RTX 5090 class Blackwell devices (sm_120a). Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(89|120a)$")
  message(FATAL_ERROR
    "NInfer supports CMAKE_CUDA_ARCHITECTURES=89 (RTX 4090, Ada) or "
    "120a (RTX 5090, Blackwell); got '${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Architecture feature gates ------------------------------------------------
# sm_120a adds Blackwell-only execution paths: TMA bulk tensor copies, warp-specialized
# producer/consumer kernels built on setmaxnreg + mbarrier, block-scaled
# kind::mxf8f6f4 / kind::mxf4nvf4 tensor core MMA, and the NVFP4 weight codecs.
# Ada (sm_89) implements none of those, so those paths are compiled out and dispatch
# falls back to the cp.async + ldmatrix + mma.sync execution models that Ada supports
# natively (including the plain E4M3 FP8 tensor core form). The K8V4 KV cache (FP8 keys
# + NVFP4 value plane through the official CUDA 13 conversion intrinsics) runs on both
# architectures, so its gate is defined unconditionally.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  add_compile_definitions(NINFER_ARCH_SM120=1 NINFER_ENABLE_TMA=1 NINFER_ENABLE_NVFP4=1
    NINFER_ENABLE_K8V4=1)
  set(NINFER_ENABLE_NVFP4 ON)
else()
  add_compile_definitions(NINFER_ARCH_SM89=1 NINFER_ENABLE_K8V4=1)
  set(NINFER_ENABLE_NVFP4 OFF)
endif()

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI, server and perplexity evaluator" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ninfer_link=1)
  set(CMAKE_JOB_POOL_LINK ninfer_link)
endif()

# Tests also exercise product input, logging and protocol adapters.
set(NINFER_BUILD_PRODUCT_SUPPORT OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_PRODUCT_SUPPORT ON)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR
    "NInfer requires CUDA 12.8 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

# --- Host toolchain tuning ----------------------------------------------------
if(MSVC)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
  add_compile_options(
    # MSVC Host C/C++. /utf-8 forces UTF-8 source and execution charset: spdlog's bundled
    # fmt static-asserts Unicode support without it, and vendored sources contain UTF-8
    # string literals that code-page decoding would corrupt.
    $<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:C,CXX>:/EHsc>
    $<$<COMPILE_LANGUAGE:C,CXX>:/O2>
    $<$<COMPILE_LANGUAGE:C,CXX>:/Ob3>
    $<$<COMPILE_LANGUAGE:C,CXX>:/Oi>
    $<$<COMPILE_LANGUAGE:C,CXX>:/Ot>
    $<$<COMPILE_LANGUAGE:C,CXX>:/arch:AVX2>

    # NVCC Host Pass-through
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Ob3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Oi>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Ot>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/arch:AVX2>
  )
  # The static CUDA runtime drags in the static CRT (libcmt), which conflicts with
  # the dynamic CRT used by the product binaries; drop the static copy at link time.
  add_link_options("LINKER:/NODEFAULTLIB:LIBCMT")
endif()

# ptxas resource reporting: host-toolchain independent and has no codegen effect, so it
# applies on every platform. Exact register counts and spill reports are the baseline
# for per-architecture kernel tuning.
add_compile_options(
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
  $<$<COMPILE_LANGUAGE:CUDA>:--split-compile=0>
)

include(cmake/Dependencies.cmake)
include(cmake/NinferTargets.cmake)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
