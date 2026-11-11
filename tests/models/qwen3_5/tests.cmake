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

# port (sm_89): the dense counterpart of the 35B fixture - qwen3.8-27b
# (architectures = ["Qwen3_5ForCausalLM"], 64 dense layers, text + vision + mtp + dflash2 in the
# component table). It pins the construction contract (max_context 4096, KV Int8Group64, MTP-3),
# the load/memory summaries against the artifact's own values, text generation at both ends of the
# 1..5 token band, append/prefix reuse, and - with NINFER_TEST_VISION=1 - a PPM image through the
# public Engine, asserting the startup-frozen vision workspace handoff.
#
# Run (from the build directory; the artifact is never hard-coded):
#   $env:NINFER_TEST_ARTIFACT = 'R:\model\qwen3_8_27b_v3\qwen3_8_27b_v3.ninfer'
#   ctest -R ninfer_qwen3_5_dense_real_test --output-on-failure
#   $env:NINFER_TEST_VISION = '1'   # the vision arm, same artifact
#   ctest -R ninfer_qwen3_5_dense_real_test --output-on-failure
# Without NINFER_TEST_ARTIFACT the binary prints "skip: NINFER_TEST_ARTIFACT is not set" and exits
# 77, which this registration maps to SKIP_RETURN_CODE.
ninfer_add_test(ninfer_qwen3_5_dense_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dense_real.cpp"
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

# port (sm_89): A1 - the routed-expert stream in Qwen4Exp's gate/up/down form. The three-bank shape
# is built synthetically (q4 gate, q4 up, q6 down, one object per expert, no artifact) and the test
# asserts the placement into all three pool runs, the per-bank tables and masks, the per-bank stride
# pricing and the eviction a pool that cannot hold every expert has to perform.
ninfer_add_test(ninfer_qwen3_5_expert_stream_split_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_expert_stream_split.cpp"
  LIBRARIES ninfer_model_runtime ninfer_core)
set_tests_properties(ninfer_qwen3_5_expert_stream_split_test PROPERTIES
  LABELS "gpu"
  RUN_SERIAL TRUE)

# port (sm_89): the gate/up/down form's LAYERED execution, numerically: a pool that holds only part
# of a layer plus a pinned tier for part of the rest, the CPU half for what neither leg can hold, and
# the per-run placement tables - all against an FP64 decode of the same bytes.
ninfer_add_test(ninfer_qwen3_5_expert_stream_split_layers_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_expert_stream_split_layers.cpp"
  LIBRARIES ninfer_model_runtime ninfer_ops ninfer_core)
set_tests_properties(ninfer_qwen3_5_expert_stream_split_layers_test PROPERTIES
  LABELS "gpu"
  RUN_SERIAL TRUE)

# v8 13.5 (step 5): runtime pool / KV reallocation under a real 4-way load. The contract: the
# idle double read refuses a resize while requests are in flight, the frozen-graph pin refuses
# at construction, and a grow->shrink round trip at a request boundary leaves the pool
# producing the control run's tokens verbatim. Cross-tier token identity is asserted only
# between same-tier batches (a wider tier shifts the host-tier split by design).
ninfer_add_test(ninfer_qwen3_5_resize_under_load_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_resize_under_load.cpp"
  LIBRARIES ninfer_engine)

# Open #1 (2026-11-05) acceptance: the hand-off race. Four clients submit back-to-back while two
# hammer threads ask for a REAL shrink (edge - 8 page groups); every attempt must be refused and
# the workload must finish intact. The engine's own `[resize]` verdict lines carry the state that
# refused each attempt.
ninfer_add_test(ninfer_qwen3_5_resize_handoff_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_resize_handoff.cpp"
  LIBRARIES ninfer_engine)

# The frozen-graph arm of the same acceptance: NINFER_GRAPH_BUDGET pays the capture allowance so# the pool is pinned at construction, and the binary answers only its own scenario so the budget
# never leaks into the load arms (the sources read the variable once per process).
add_test(NAME ninfer_qwen3_5_resize_pinned_graph_real_test
  COMMAND ninfer_qwen3_5_resize_under_load_real_test)
# NINFER_LAYER_GRAPH=0 because the per-layer capture defaults on and its replayed rounds call a
# stream sync the capture forbids (ffn.cpp:355, cudaErrorStreamCaptureUnsupported) once the
# budget is paid on this artifact; the pool pin and the resize refusal do not depend on it.
set_tests_properties(ninfer_qwen3_5_resize_pinned_graph_real_test PROPERTIES
  ENVIRONMENT "NINFER_RESIZE_TEST_SCENARIO=pinned;NINFER_GRAPH_BUDGET=1;NINFER_LAYER_GRAPH=0")

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

# T131: the control plane's pure half - the three console modes and the freeze family's ranges.
# Also host-only: the mapping is a header, and the engine-side park/apply half is covered by the
# serve acceptance plus the live run.
ninfer_add_test(ninfer_qwen3_5_control_plane_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_control_plane.cpp"
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
  ninfer_qwen3_5_dense_real_test
  ninfer_qwen3_5_dflash_real_test
  ninfer_qwen3_5_kv_tier_real_test
  ninfer_qwen3_5_expert_stream_real_test
  ninfer_qwen3_5_resize_under_load_real_test
  ninfer_qwen3_5_resize_handoff_real_test
  ninfer_qwen3_5_resize_pinned_graph_real_test)
set_tests_properties(${ninfer_qwen3_5_real_tests} PROPERTIES
  SKIP_RETURN_CODE 77
  RUN_SERIAL TRUE
  LABELS "gpu;real")

# The resize acceptance builds three real engines and runs six 4-way batches; the default
# ctest timeout would kill it long before the pinned-graph engine exists.
set_tests_properties(ninfer_qwen3_5_resize_under_load_real_test PROPERTIES
  TIMEOUT 5400)
set_tests_properties(ninfer_qwen3_5_resize_pinned_graph_real_test PROPERTIES
  TIMEOUT 1800)
set_tests_properties(ninfer_qwen3_5_resize_handoff_real_test PROPERTIES
  TIMEOUT 1800)
# A 19 GiB resident load plus the vision arm; the default timeout is not enough for the upload alone.
# port (sm_89): the fixture now also carries the 256K residency arm, which builds a second 19 GiB
# engine in a process of its own (see the registration below), hence the larger ceiling.
set_tests_properties(ninfer_qwen3_5_dense_real_test PROPERTIES
  TIMEOUT 5400)

# port (sm_89): the 256K arm of the same fixture - "logical KV capacity above what the Device can
# hold resident, the rest on the Host". It is a separate ctest entry because it must own a fresh
# process: two 19 GiB engines back to back left the second one seeing "0 bytes free" on WDDM even
# though the first had been destroyed. Its own budget: one construction (~2 min), the one-token
# contract, the overrun refusal (which must return in bounded time - measured inside the test) and,
# with NINFER_TEST_256K_BENCH, a 32K-token decode. The 1800 s ceiling is deliberately tighter than
# "an hour later": the overrun request used to hang forever, and this entry has to fail loudly.
add_test(NAME ninfer_qwen3_5_dense_256k_residency_test
  COMMAND ninfer_qwen3_5_dense_real_test)
set_tests_properties(ninfer_qwen3_5_dense_256k_residency_test PROPERTIES
  ENVIRONMENT "NINFER_TEST_DENSE_ARM=256k"
  SKIP_RETURN_CODE 77
  RUN_SERIAL TRUE
  LABELS "gpu;real"
  TIMEOUT 1800)

# port (sm_89): the swap arm of the same fixture. It is the only scenario in which the Host KV tier
# actually does work under residency: R1 retains a 4K-token prefix, R2 prefills almost the whole
# resident set (which forces the retained prefix onto the Host tier), and R3 reuses the prefix so the
# binding has to fetch it back. Asserts fetched_pages > 0, host_occupied > 0 and R3 == R1 token for
# token. Its own process and its own budget: one construction (~2 min) plus ~25 s of prefills.
add_test(NAME ninfer_qwen3_5_dense_256k_swap_test
  COMMAND ninfer_qwen3_5_dense_real_test)
set_tests_properties(ninfer_qwen3_5_dense_256k_swap_test PROPERTIES
  ENVIRONMENT "NINFER_TEST_DENSE_ARM=256k;NINFER_TEST_256K_SWAP=1"
  SKIP_RETURN_CODE 77
  RUN_SERIAL TRUE
  LABELS "gpu;real"
  TIMEOUT 1800)

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

# S1 of the QSA wiring: the index plane, the tail state pool and the two stateful ops' scratch are
# sized from one place, and the transient side is delegated to the ops' own capacity queries so a
# controller dial that widens the interval can never leave the plan behind execution.
ninfer_add_test(ninfer_qsa_plan_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_qsa_plan.cpp"
  LIBRARIES ninfer_model_runtime ninfer_ops)
