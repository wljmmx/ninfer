ninfer_add_test(ninfer_qwen4_exp_config_bindings_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_config_bindings.cpp"
  LIBRARIES ninfer_model_runtime)

ninfer_add_test(ninfer_ple_disk_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ple_disk.cpp"
  LIBRARIES ninfer_ple ninfer::json)

# port (sm_89): the q4_g64 table encoding the shipped artifact declares. The host row decoder is
# checked word for word against the shipped q4 kernel (a one-hot linear) and against the artifact
# codecs' independent logical decoder, then end to end through the disk table's two-row planes.
ninfer_add_test(ninfer_qwen4_exp_ple_q4_row_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ple_q4_row.cpp"
  LIBRARIES ninfer_ple ninfer_ops ninfer_linear_test_support ninfer::json)
set_tests_properties(ninfer_qwen4_exp_ple_q4_row_test PROPERTIES
  SKIP_RETURN_CODE 77
  LABELS "gpu")

# port (sm_89): the PLE layer end to end at the artifact's geometry - one [12800,2560] projection,
# the per-stream gate, the four-tap convolution folded into the residual, and the history pool that
# advances by the accepted prefix. Numeric gates reuse the Ops' own criteria.
ninfer_add_test(ninfer_qwen4_exp_ple_layer_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ple_layer.cpp"
  LIBRARIES ninfer_ple ninfer_ops)
set_tests_properties(ninfer_qwen4_exp_ple_layer_test PROPERTIES
  SKIP_RETURN_CODE 77
  LABELS "gpu")

# port (sm_89): PLE wired into the text block. A two-block stack whose second block carries PLE,
# driven through the owner text.cpp calls, with the embedding gathered from a real NF4 table on
# disk; the off path (no PLE block) is the baseline it compares word for word.
ninfer_add_test(ninfer_qwen4_exp_ple_block_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ple_block.cpp"
  LIBRARIES ninfer_ple ninfer_ops ninfer::json)
set_tests_properties(ninfer_qwen4_exp_ple_block_test PROPERTIES
  SKIP_RETURN_CODE 77
  LABELS "gpu")

ninfer_add_test(ninfer_qwen4_exp_parameters_banked_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_parameters_banked.cpp"
  LIBRARIES ninfer_model_runtime)

ninfer_add_test(ninfer_qwen4_exp_projection_dispatch_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_projection_dispatch.cpp"
  LIBRARIES ninfer_model_runtime)

# port (sm_89): A2 - one Qwen4Exp MoE layer end to end through ffn(). The routed term over three
# banks (gate/up each I rows, down [H,I]), the shared expert as its own Q5 projection pair, and the
# two residual seams that form implies (routed into the epilogue, shared through scaled_add).
ninfer_add_test(ninfer_qwen4_exp_moe_split_layer_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_moe_split_layer.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_qwen4_exp_moe_split_layer_test PROPERTIES
  SKIP_RETURN_CODE 77
  LABELS "gpu")
