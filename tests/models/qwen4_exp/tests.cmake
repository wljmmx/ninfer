ninfer_add_test(ninfer_qwen4_exp_config_bindings_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_config_bindings.cpp"
  LIBRARIES ninfer_model_runtime)

ninfer_add_test(ninfer_ple_disk_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ple_disk.cpp"
  LIBRARIES ninfer_ple ninfer::json)

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
