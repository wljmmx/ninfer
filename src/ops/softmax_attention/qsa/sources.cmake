target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/input_projection.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/input_projection_bf16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/input_projection_q8.cu"
)
