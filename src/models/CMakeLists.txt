# Immutable loading and mutable execution have separate compile owners.
add_library(ninfer_model_loading STATIC registry.cpp)
ninfer_internal_includes(ninfer_model_loading)
target_link_libraries(ninfer_model_loading PUBLIC ninfer_artifact ninfer_text ninfer_ops ninfer_ple)

add_library(ninfer_model_runtime STATIC)
ninfer_internal_includes(ninfer_model_runtime)
target_link_libraries(ninfer_model_runtime
  PUBLIC ninfer_model_loading ninfer_ops ninfer_runtime_support
  # The Qwen4Exp text path drives PLE from run_layers (text.cpp), so the disk layer's window
  # owner is a link dependency of the runtime, not only of the PLE tests.
  PRIVATE ninfer_ple ninfer_media_decode ninfer_jinja ninfer_grammar ninfer_xgrammar)
target_include_directories(ninfer_model_runtime PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/xgrammar/cpp)
target_include_directories(ninfer_model_runtime SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/xgrammar/3rdparty/picojson)

include(qwen3_5/loading_sources.cmake)
include(qwen3_5/frontend_sources.cmake)
include(qwen3_5/execution_sources.cmake)
include(qwen3_5/program_sources.cmake)

# port (sm_89): qwen4-exp PLE disk layer. The table/row-reader/prepare pipeline is CPU only;
# the optional GPU staging owner still links CUDA through ninfer_core. Upstream reads through
# io_uring (core/io) - this tree routes every row read through artifact::InputFile's
# NO_BUFFERING overlapped reads instead.
add_library(ninfer_ple STATIC
  qwen4_exp/ple/table.cpp
  qwen4_exp/ple/hash.cpp
  qwen4_exp/ple/read_queue.cpp
  qwen4_exp/ple/row_reader.cpp
  qwen4_exp/ple/prepare.cpp
  qwen4_exp/ple/gpu.cpp
  qwen4_exp/execution/ple_layer.cpp
)
ninfer_internal_includes(ninfer_ple)
target_link_libraries(ninfer_ple PUBLIC ninfer_artifact ninfer_core ninfer_ops)
