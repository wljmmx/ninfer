# The family's own execution sources. Nothing here is reachable from a Qwen3.5 artifact: the QSA
# block selection and the four-stream residual are this family's structure, and its registry
# identity is what turns them on.
target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/execution/qsa.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/qsa_layer.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/hc.cpp"
)
