ninfer_add_test(ninfer_qwen3_5_loading_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading_real.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen3_5_loading_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen3_5_frontend_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core ninfer::json)

ninfer_add_test(ninfer_qwen3_5_runtime_mechanisms_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_runtime_mechanisms.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_state_image_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_state_image_layout_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image_layout.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_context_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_context_store.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_native_transactions_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_native_transactions.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen3_5_prefix_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_prefix_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_preemption_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_preemption_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_grammar_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_grammar_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_tools_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_tools_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_score_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_score_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_vision_workspace_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_workspace.cpp"
  LIBRARIES ninfer_model_runtime ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_dflash2_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash2_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_dflash_prefill_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dflash_prefill_real.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen3_5_moe_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_moe_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_dflash_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash_real.cpp"
  LIBRARIES ninfer_engine)

# port (sm_89): Q54's prefix index against the real artifact: exact-prompt restores, the two
# long-prefix tiers (500 / 4100), the whole-prompt decline, and the no-index / bad-index
# fallbacks - every arm compared token for token against a cold run of the same prompt.
ninfer_add_test(ninfer_qwen3_5_kv_tier_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_kv_tier_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_expert_stream_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_expert_stream_real.cpp"
  LIBRARIES ninfer_artifact ninfer_ops ninfer_core)

ninfer_add_test(ninfer_tool_call_parser_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../../test_tool_call_parser.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_visual_scatter_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_visual_scatter.cpp"
  LIBRARIES ninfer_engine ninfer_core)

# v8 section 12, first phase: the parameter bus, the serial rule (9.3) and the three deployment
# modes are pure host-side logic - no GPU, no model - so this runs in the plain test list.
ninfer_add_test(ninfer_qwen3_5_param_bus_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_param_bus.cpp"
  LIBRARIES ninfer_core)

add_test(NAME ninfer_qwen3_5_agent_continuation_real_test
  COMMAND ninfer_qwen3_5_prefix_real_test)
set_tests_properties(ninfer_qwen3_5_agent_continuation_real_test PROPERTIES
  ENVIRONMENT "NINFER_PREFIX_REAL_SCENARIO=agent-continuation")

# The merged prefill pass's tail-column addressing: two rows with different prompts and lengths
# must produce the same greedy tokens as the single-lane path, and the captured dispatch log has
# to prove the two-row pass actually ran.
add_test(NAME ninfer_qwen3_5_merged_prefill_real_test
  COMMAND ninfer_qwen3_5_prefix_real_test)
set_tests_properties(ninfer_qwen3_5_merged_prefill_real_test PROPERTIES
  ENVIRONMENT "NINFER_PREFIX_REAL_SCENARIO=merged-prefill")

# The production-shaped merge: four lanes at the 256-column class and four 3328-token prompts at
# chunk 1024, both arms admitted together so only the merged pass differs, with the four-row pass
# pinned in the dispatch log and the per-row begin token asserted separately from the full
# sequence (the dense decode is scheduled per batch size, a pre-existing composition window).
add_test(NAME ninfer_qwen3_5_merged_prefill_production_real_test
  COMMAND ninfer_qwen3_5_prefix_real_test)
set_tests_properties(ninfer_qwen3_5_merged_prefill_production_real_test PROPERTIES
  ENVIRONMENT "NINFER_PREFIX_REAL_SCENARIO=merged-prefill-production")

# A real-model test owns the single GPU while its artifact is resident.
set(ninfer_qwen3_5_real_tests
  ninfer_qwen3_5_loading_real_test
  ninfer_qwen3_5_native_transactions_test
  ninfer_qwen3_5_prefix_real_test
  ninfer_qwen3_5_agent_continuation_real_test
  ninfer_qwen3_5_merged_prefill_real_test
  ninfer_qwen3_5_merged_prefill_production_real_test
  ninfer_qwen3_5_preemption_real_test
  ninfer_qwen3_5_grammar_real_test
  ninfer_qwen3_5_tools_real_test
  ninfer_qwen3_5_score_real_test
  ninfer_qwen3_5_vision_workspace_test
  ninfer_qwen3_5_dflash2_real_test
  ninfer_qwen3_5_dflash_prefill_real_test
  ninfer_qwen3_5_moe_real_test
  ninfer_qwen3_5_dflash_real_test
  ninfer_qwen3_5_kv_tier_real_test
  ninfer_qwen3_5_expert_stream_real_test)
set_tests_properties(${ninfer_qwen3_5_real_tests} PROPERTIES
  SKIP_RETURN_CODE 77
  RUN_SERIAL TRUE
  LABELS "gpu;real")

set_tests_properties(
  ninfer_qwen3_5_state_image_test
  ninfer_qwen3_5_context_store_test
  ninfer_qwen3_5_visual_scatter_test
  PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")

ninfer_add_test(ninfer_qwen3_5_tool_constraints_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tool_constraints.cpp"
  LIBRARIES ninfer_model_runtime ninfer_grammar ninfer::json)

add_test(NAME ninfer_qwen3_5_tool_schema_oracle_test
  COMMAND ${CMAKE_COMMAND} -E env
    "NINFER_TOOL_PROBE=$<TARGET_FILE:ninfer_qwen3_5_tool_constraints_test>"
    ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/test_tool_schema.py")
