add_library(ninfer_linear_add_test_support STATIC
  "${CMAKE_CURRENT_LIST_DIR}/linear_add_test_common.cpp")
ninfer_test_includes(ninfer_linear_add_test_support)
ninfer_op_oracle_options(ninfer_linear_add_test_support)
target_link_libraries(ninfer_linear_add_test_support PUBLIC ninfer_ops)

ninfer_add_op_test(ninfer_linear_add_q4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a16.cpp"
  LIBRARIES ninfer_linear_add_test_support)

ninfer_add_op_test(ninfer_linear_add_q5_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q5_a16.cpp"
  LIBRARIES ninfer_linear_add_test_support)

ninfer_add_op_test(ninfer_linear_add_bf16_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_bf16_a16.cpp"
  LIBRARIES ninfer_linear_add_test_support)

ninfer_add_op_test(ninfer_linear_add_q8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a16.cpp"
  LIBRARIES ninfer_linear_add_test_support)

# NVFP4 kernels exist only in Blackwell (sm_120a) builds; the sm_89 dispatch rejects
# NVFP4 weights, so this correctness test is registered only when the kernels exist.
if(NINFER_ENABLE_NVFP4)
  ninfer_add_op_test(ninfer_linear_add_nvfp4_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4.cpp"
    LIBRARIES ninfer_ops)
endif()

ninfer_add_op_test(ninfer_linear_add_fp8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8.cpp"
  LIBRARIES ninfer_ops)
