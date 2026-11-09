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
