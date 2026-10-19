# port: upstream's Linux implementation is kept and a Win32 one is selected on Windows.
if(WIN32)
  set(NINFER_CPU_PLATFORM_SOURCES cpu/system_windows.cpp)
  set(NINFER_CPU_PLATFORM_LIBRARIES Threads::Threads)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
       CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(NINFER_CPU_PLATFORM_SOURCES cpu/system.cpp)
  set(NINFER_CPU_PLATFORM_LIBRARIES ${NINFER_HWLOC_TARGET} Threads::Threads)
else()
  message(FATAL_ERROR "NInfer CPU execution supports Linux x86-64 and Windows x86-64")
endif()

add_library(ninfer_cpu STATIC
  ${NINFER_CPU_PLATFORM_SOURCES}
  cpu/task_pool.cpp
  cpu/executor.cpp
)
ninfer_internal_includes(ninfer_cpu)
target_link_libraries(ninfer_cpu PRIVATE ${NINFER_CPU_PLATFORM_LIBRARIES})

# port: Win32 IO sources replace upstream's io_uring implementation on Windows.
if(WIN32)
  set(NINFER_IO_PLATFORM_SOURCES io/file_windows.cpp io/read_queue_windows.cpp io/device_traits.cpp)
else()
  set(NINFER_IO_PLATFORM_SOURCES io/file.cpp io/read_queue.cpp io/device_traits.cpp)
endif()

add_library(ninfer_core STATIC
  arena.cpp
  buffer.cpp
  buffer.cu
  memory.cpp
  transfer.cpp
  transfer.cu
  device.cu
  dtype.cpp
  nf4.cpp
  ${NINFER_IO_PLATFORM_SOURCES}
  gdn_replay_records.cpp
  cyclic_kv_cache.cpp
  host_context_arena.cpp
  host_kv_arena.cpp
  paged_kv_cache.cpp
  linear_attention_state.cpp
  state_image.cpp
  layout.cpp
  tensor.cpp
  weight_view.cpp
  weight_bank.cpp
  expert_cache.cpp
  host_reserve.cpp
  vmm_pool.cpp
  decode_graph.cpp
)
ninfer_internal_includes(ninfer_core)
ninfer_cuda_archive(ninfer_core)
target_link_libraries(ninfer_core PUBLIC ninfer_cpu CUDA::cudart CUDA::nvtx3 Threads::Threads)
# port (sm_89): VmmPool uses the driver VMM entry points.
target_link_libraries(ninfer_core PUBLIC CUDA::cuda_driver)
if(NOT WIN32)
  target_link_libraries(ninfer_core PRIVATE ${NINFER_URING_TARGET})
endif()
