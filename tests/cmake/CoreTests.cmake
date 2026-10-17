# This target deliberately receives no src/, CUDA, artifact, kernel, or target
# include root. It proves that the public product headers stand alone.
add_executable(ninfer_public_api_test "${CMAKE_CURRENT_LIST_DIR}/../test_public_api.cpp")
ninfer_deploy_runtime_dlls(ninfer_public_api_test)
target_include_directories(ninfer_public_api_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME ninfer_public_api_test COMMAND ninfer_public_api_test)

foreach(part IN ITEMS system task_pool executor)
  ninfer_add_test(ninfer_cpu_${part}_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/../cpu/test_${part}.cpp"
    LIBRARIES ninfer_cpu)
  set_tests_properties(ninfer_cpu_${part}_test PROPERTIES TIMEOUT 30)
endforeach()

ninfer_add_test(ninfer_device_test       SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_device.cpp"
  LIBRARIES ninfer_core)

set_tests_properties(ninfer_device_test PROPERTIES
  ENVIRONMENT_MODIFICATION "NINFER_CUDA_SYNC=unset:")
set(sync_modes spin blocking yield auto)
set(sync_flags 1 4 2 0)
foreach(mode flags IN ZIP_LISTS sync_modes sync_flags)
  add_test(NAME ninfer_device_sync_${mode}_test COMMAND ninfer_device_test ${flags})
  set_tests_properties(ninfer_device_sync_${mode}_test PROPERTIES
    ENVIRONMENT "NINFER_CUDA_SYNC=${mode}" SKIP_RETURN_CODE 77)
endforeach()
foreach(mode IN ITEMS invalid empty)
  add_test(NAME ninfer_device_sync_${mode}_test COMMAND ninfer_device_test --invalid-sync)
endforeach()
set_tests_properties(ninfer_device_sync_invalid_test PROPERTIES
  ENVIRONMENT "NINFER_CUDA_SYNC=invalid")
set_tests_properties(ninfer_device_sync_empty_test PROPERTIES
  ENVIRONMENT "NINFER_CUDA_SYNC=")

ninfer_add_test(ninfer_decode_graph_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_decode_graph.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_tensor_test       SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_tensor.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_memory_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_memory.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_nf4_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_nf4.cpp"
  LIBRARIES ninfer_core)
add_test(NAME ninfer_nf4_exact_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/../test_nf4_exact.py"
    $<TARGET_FILE:ninfer_nf4_test>)

# port (sm_89): the io_uring read queue does not exist on Windows (the port ships an IOCP
# implementation in core/io/read_queue_windows.cpp) and the oracle needs GNU ld --wrap=io_uring_submit,
# which MSVC cannot express. Windows read-queue coverage is a known gap; the source stays in the tree.
if(NOT WIN32)
  ninfer_add_test(ninfer_read_queue_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../io/test_read_queue.cpp"
    LIBRARIES ninfer_core ${NINFER_URING_TARGET})
  target_link_options(ninfer_read_queue_test PRIVATE "LINKER:--wrap=io_uring_submit")
  set_tests_properties(ninfer_read_queue_test PROPERTIES TIMEOUT 30)
else()
  # The IOCP build has no io_uring_submit to wrap, so this test pins the contract instead: window
  # arithmetic, byte-exact reads at unaligned offsets, draining, buffer reuse across rounds (the
  # release-in-poll regression) and the EOF tail.
  ninfer_add_test(ninfer_read_queue_windows_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/../io/test_read_queue_windows.cpp"
    LIBRARIES ninfer_core)
  set_tests_properties(ninfer_read_queue_windows_test PROPERTIES TIMEOUT 30)
endif()

ninfer_add_test(ninfer_arena_test        SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_arena.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_kv_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_kv_cache.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_host_context_arena_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_host_context_arena.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_state_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_state_store.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_gdn_replay_records_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_gdn_replay_records.cpp"
  LIBRARIES ninfer_core)

set_tests_properties(
  ninfer_device_test
  ninfer_decode_graph_test
  ninfer_arena_test
  ninfer_kv_cache_test
  ninfer_host_context_arena_test
  ninfer_state_store_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_host_timing_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_host_timing.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_jinja_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../text/test_jinja.cpp"
  LIBRARIES ninfer_jinja ninfer::json)

add_test(NAME ninfer_chat_templates_test
  COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tests/text/test_chat_templates.py
          $<TARGET_FILE:ninfer_jinja_test>)

ninfer_add_test(ninfer_weight_bank_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_weight_bank.cpp"
  LIBRARIES ninfer_core)

# port (sm_89): the failure oracle replaces cuda* at link time with __wrap_/__real_ pairs. Only
# GNU ld can interpose a symbol that arrives through an import library, so on Windows the oracle TU
# is left out and the test skips its injection section (NINFER_HAS_CUDA_FAULT_WRAP). Everything else
# in this test runs unmodified.
set(NINFER_EXPERT_CACHE_TEST_SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_expert_cache.cu")
if(NOT WIN32)
  list(APPEND NINFER_EXPERT_CACHE_TEST_SOURCES
       "${CMAKE_CURRENT_LIST_DIR}/../expert_cache_cuda_errors.cpp")
endif()
ninfer_add_test(ninfer_expert_cache_test
  SOURCES ${NINFER_EXPERT_CACHE_TEST_SOURCES}
  LIBRARIES ninfer_core)
set_tests_properties(ninfer_expert_cache_test PROPERTIES TIMEOUT 60)

if(NOT WIN32)
  foreach(symbol IN ITEMS
      cudaEventRecord cudaMemcpyAsync cudaMemcpyBatchAsync cudaStreamSynchronize cudaStreamWaitEvent
      cudaFree cudaFreeHost)
    target_link_options(ninfer_expert_cache_test PRIVATE "LINKER:--wrap=${symbol}")
  endforeach()
  target_compile_definitions(ninfer_expert_cache_test PRIVATE NINFER_HAS_CUDA_FAULT_WRAP=1)
endif()
