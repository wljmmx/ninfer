target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/rmsnorm_rope.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/rmsnorm_rope_2d.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/rmsnorm_rope_2d.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launch.cu"
)
