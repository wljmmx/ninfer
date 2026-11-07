# Build-target selection, adapted from upstream's separate per-device targets
# (upstream 8ed68325 splits rtx5090 and dgx_spark). This tree ships the two
# single-GPU Windows targets instead: RTX 4090 class Ada (sm_89) and RTX 5090
# class Blackwell (sm_120a). There is no DGX Spark target here.

# Honour an explicit architecture so existing configure lines keep working.
if(NOT DEFINED NINFER_TARGET)
  if(DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
    set(_ninfer_target_from_arch rtx5090)
  else()
    set(_ninfer_target_from_arch rtx4090)
  endif()
  set(NINFER_TARGET ${_ninfer_target_from_arch} CACHE STRING "NInfer execution target")
  unset(_ninfer_target_from_arch)
endif()
set_property(CACHE NINFER_TARGET PROPERTY STRINGS rtx4090 rtx5090)

if(NINFER_TARGET STREQUAL "rtx4090")
  set(NINFER_CUDA_ARCHITECTURE 89)
elseif(NINFER_TARGET STREQUAL "rtx5090")
  set(NINFER_CUDA_ARCHITECTURE 120a)
else()
  message(FATAL_ERROR
    "Unknown NINFER_TARGET '${NINFER_TARGET}'; NInfer supports rtx4090 (sm_89, Ada) "
    "and rtx5090 (sm_120a, Blackwell)")
endif()

# CUDA compiler detection must see the selected architecture, not a host default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES ${NINFER_CUDA_ARCHITECTURE}
    CACHE STRING "CUDA architectures to build")
endif()

function(ninfer_validate_cuda_architecture)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL NINFER_CUDA_ARCHITECTURE)
    message(FATAL_ERROR
      "NINFER_TARGET=${NINFER_TARGET} requires CMAKE_CUDA_ARCHITECTURES="
      "${NINFER_CUDA_ARCHITECTURE}; got '${CMAKE_CUDA_ARCHITECTURES}'")
  endif()
endfunction()

# One build directory compiles one target: sm_89 and sm_120a objects must not be
# mixed in a single tree. Toolchain files can also change the architecture during
# project() initialization, so the check runs again after project().
function(ninfer_validate_target)
  if(DEFINED NINFER_CONFIGURED_TARGET AND
     NOT NINFER_CONFIGURED_TARGET STREQUAL NINFER_TARGET)
    message(FATAL_ERROR
      "This build directory was configured for ${NINFER_CONFIGURED_TARGET}; "
      "use a separate build directory for ${NINFER_TARGET}")
  endif()
  ninfer_validate_cuda_architecture()
  set(NINFER_CONFIGURED_TARGET ${NINFER_TARGET} CACHE INTERNAL "Configured NInfer target")
  message(STATUS "NInfer target: ${NINFER_TARGET} (sm_${NINFER_CUDA_ARCHITECTURE})")
endfunction()
