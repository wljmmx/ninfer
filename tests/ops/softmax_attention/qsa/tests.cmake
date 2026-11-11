ninfer_add_op_test(ninfer_qsa_input_proj_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_input_projection.cpp"
  LIBRARIES ninfer_linear_test_support)

# The real Qwen4Exp artifact stores this layer's projections as Q4/Q5 row regions of shared
# parents, which the fused Op above does not register. Its separate-row route is pinned here.
ninfer_add_op_test(ninfer_qsa_input_proj_q4_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_input_projection_q4.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_qsa_tail_extract_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tail_extract.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_qsa_prepare_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prepare.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_qsa_indexer_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_indexer.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_qsa_sparse_gqa_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sparse_gqa.cpp"
  LIBRARIES ninfer_ops)

# The merged batch's row equivalence: both arms are built with the same (tokens, batch, envelope)
# so sparse_gqa_plan resolves one plan for both; the single-row arm's other columns are inert.
ninfer_add_op_test(ninfer_qsa_batch_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_qsa_batch.cpp"
  LIBRARIES ninfer_ops)
