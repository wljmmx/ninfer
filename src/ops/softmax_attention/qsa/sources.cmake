target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/input_projection.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/input_projection_bf16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/input_projection_q8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/tail_extract.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/tail_extract.cu"
  "${CMAKE_CURRENT_LIST_DIR}/prepare.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/prepare_launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/indexer.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/indexer_launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sparse_gqa.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sparse_gqa_launch.cu"
)
