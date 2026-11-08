target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/gdn_control.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execute.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/control.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/n96_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/n64_k2048.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/n96_k2560.cu")
