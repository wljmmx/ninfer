# The local tree keeps the pre-reorganization RMSNorm family in
# src/ops/{kernel,launcher,wrapper}; these two fused forms arrive with the
# post-reorganization layout and keep their upstream paths. Upstream builds the
# launch .cu units into ninfer_ops_non_rdc; the local tree has no such target and
# these kernels use no RDC-only feature, so they join ninfer_ops directly.
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/paired_concat.cu"
  "${CMAKE_CURRENT_LIST_DIR}/residual_add.cu"
  "${CMAKE_CURRENT_LIST_DIR}/paired_concat.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/residual_add.cpp")
