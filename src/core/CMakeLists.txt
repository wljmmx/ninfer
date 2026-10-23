# port (sm_89): the bench CPU execution layer. Upstream implements the topology and affinity
# API with hwloc; the Win32 implementation keeps the same header and semantics, which is the
# only platform this tree ships. Linux support is left to the bench tree.
add_library(ninfer_cpu STATIC
  cpu/system_windows.cpp
  cpu/task_pool.cpp
  cpu/executor.cpp
)
ninfer_internal_includes(ninfer_cpu)
target_link_libraries(ninfer_cpu PRIVATE Threads::Threads)

add_library(ninfer_core STATIC
  arena.cu
cuda_host.cpp
memory.cpp
memory_arena.cpp
  expert_cache.cpp
  vmm_pool.cpp
weight_bank.cpp
  device.cu
  dtype.cpp
  host_worker_pool.cpp
  gdn_replay_records.cpp
  cyclic_kv_cache.cpp
  host_context_arena.cpp
  host_kv_arena.cpp
  kv_spill_file.cpp
  paged_kv_cache.cpp
  linear_attention_state.cpp
  layout.cpp
  tensor.cpp
  weight_view.cpp
  decode_graph.cpp
)
ninfer_internal_includes(ninfer_core)
ninfer_cuda_archive(ninfer_core)
target_link_libraries(ninfer_core PUBLIC ninfer_cpu CUDA::cudart CUDA::nvtx3 Threads::Threads)
# port (sm_89): VmmPool drives the driver VMM entry points directly.
target_link_libraries(ninfer_core PUBLIC CUDA::cuda_driver)

