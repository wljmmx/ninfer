target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/gated_residual.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/mix/mix.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/mix/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/mix/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/mix/launch.cu")
