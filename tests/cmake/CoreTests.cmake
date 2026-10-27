# This target deliberately receives no src/, CUDA, artifact, kernel, or target
# include root. It proves that the public product headers stand alone.
add_executable(ninfer_public_api_test "${CMAKE_CURRENT_LIST_DIR}/../test_public_api.cpp")
target_include_directories(ninfer_public_api_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME ninfer_public_api_test COMMAND ninfer_public_api_test)

ninfer_add_test(ninfer_device_test       SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_device.cpp"
  LIBRARIES ninfer_core)

set_tests_properties(ninfer_device_test PROPERTIES
  ENVIRONMENT_MODIFICATION "NINFER_CUDA_SYNC=unset:")
set(sync_modes spin blocking yield auto)
set(sync_flags 1 4 2 0)
foreach(mode flags IN ZIP_LISTS sync_modes sync_flags)
  add_test(NAME ninfer_device_sync_${mode}_test COMMAND ninfer_device_test ${flags})
  # ENVIRONMENT_MODIFICATION (not ENVIRONMENT) is per test and restored afterwards.
  # The legacy ENVIRONMENT form leaves the empty value set for every following test,
  # and an empty NINFER_CUDA_SYNC is not a valid mode, so device init throws in all of
  # them; that made unrelated tests fail purely on suite order.
  set_tests_properties(ninfer_device_sync_${mode}_test PROPERTIES
    ENVIRONMENT_MODIFICATION "NINFER_CUDA_SYNC=set:${mode}" SKIP_RETURN_CODE 77)
endforeach()
foreach(mode IN ITEMS invalid empty)
  add_test(NAME ninfer_device_sync_${mode}_test COMMAND ninfer_device_test --invalid-sync)
endforeach()
set_tests_properties(ninfer_device_sync_invalid_test PROPERTIES
  ENVIRONMENT_MODIFICATION "NINFER_CUDA_SYNC=set:invalid")
set_tests_properties(ninfer_device_sync_empty_test PROPERTIES
  ENVIRONMENT_MODIFICATION "NINFER_CUDA_SYNC=set:")

ninfer_add_test(ninfer_decode_graph_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_decode_graph.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_tensor_test       SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_tensor.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_arena_test        SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_arena.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_kv_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_kv_cache.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_host_context_arena_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_host_context_arena.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_kv_spill_file_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_kv_spill_file.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_vmm_pool_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_vmm_pool.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_expert_pool_resize_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_expert_pool_resize.cpp"
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
  ninfer_vmm_pool_test
  ninfer_expert_pool_resize_test
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

ninfer_add_test(ninfer_expert_cache_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_expert_cache.cpp" LIBRARIES ninfer_core)

# The KV tier protocol and the prefetch queue: both are header-only seams with no GPU dependency.
ninfer_add_test(ninfer_kv_seam_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/kv_seam_test.cpp"
  LIBRARIES ninfer_core)

# The RESP transport for the LMCache remote tier: encoding always runs, the round trip runs only
# when NINFER_RESP_PORT points at a Redis (77 = skipped otherwise).
ninfer_add_test(ninfer_resp_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/resp_test.cpp"
  LIBRARIES ninfer_core $<$<PLATFORM_ID:Windows>:ws2_32>)
set_tests_properties(ninfer_resp_test PROPERTIES SKIP_RETURN_CODE 77)

# P1-b: the LMCache key format, the consecutive-prefix rule, and the Redis-backed KV tier.
ninfer_add_test(ninfer_kv_tier_redis_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/kv_tier_redis_test.cpp"
  LIBRARIES ninfer_core $<$<PLATFORM_ID:Windows>:ws2_32>)
set_tests_properties(ninfer_kv_tier_redis_test PROPERTIES SKIP_RETURN_CODE 77)

# P1-b: chunk hashing (ours, chained) and the contiguous-prefix skip length.
ninfer_add_test(ninfer_kv_reuse_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/kv_reuse_test.cpp"
  LIBRARIES ninfer_core)

# P1-b-2d: the checkpoint payload (KV spans + state spans) the KV tier has to carry.
ninfer_add_test(ninfer_kv_checkpoint_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/kv_checkpoint_test.cpp"
  LIBRARIES ninfer_core)
