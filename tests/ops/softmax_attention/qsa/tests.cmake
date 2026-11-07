ninfer_add_op_test(ninfer_qsa_input_proj_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_input_projection.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_qsa_tail_extract_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tail_extract.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_qsa_prepare_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prepare.cpp"
  LIBRARIES ninfer_ops)
