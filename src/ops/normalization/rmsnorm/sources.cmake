# The fused RMSNorm+concat and residual-add forms that upstream ships under this
# path; the plain/gated RMSNorm family lives in src/ops/rmsnorm since the T104
# reorganization. Upstream builds the launch .cu units into ninfer_ops_non_rdc;
# the local tree has no such target and these kernels use no RDC-only feature,
# so they join ninfer_ops directly.
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/paired_concat.cu"
  "${CMAKE_CURRENT_LIST_DIR}/residual_add.cu"
  "${CMAKE_CURRENT_LIST_DIR}/paired_concat.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/residual_add.cpp")
