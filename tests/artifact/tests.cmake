ninfer_add_test(ninfer_artifact_reader_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_reader.cpp"
  LIBRARIES ninfer_artifact)

ninfer_add_test(ninfer_artifact_backing_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_backing.cpp"
  LIBRARIES ninfer_artifact)
set_tests_properties(ninfer_artifact_backing_test PROPERTIES TIMEOUT 30)

# port (sm_89): same story as the expert-cache oracle -- the failure injection is a GNU ld
# --wrap interposition of cuda*, which MSVC cannot express. On Windows the oracle TU is left out
# and the test skips that section (NINFER_HAS_CUDA_FAULT_WRAP); the rest runs unmodified.
set(NINFER_MATERIALIZATION_TEST_SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_materialization.cpp")
if(NOT WIN32)
  list(APPEND NINFER_MATERIALIZATION_TEST_SOURCES
       "${CMAKE_CURRENT_LIST_DIR}/materialization_cuda_errors.cpp")
endif()
ninfer_add_test(ninfer_artifact_materialization_test
  SOURCES ${NINFER_MATERIALIZATION_TEST_SOURCES}
  LIBRARIES ninfer_artifact)

if(NOT WIN32)
  target_link_options(ninfer_artifact_materialization_test PRIVATE
    "LINKER:--wrap=cudaMalloc"
    "LINKER:--wrap=cudaMallocHost"
    "LINKER:--wrap=cudaFree"
    "LINKER:--wrap=cudaFreeHost"
    "LINKER:--wrap=cudaEventCreateWithFlags"
    "LINKER:--wrap=cudaEventRecord"
    "LINKER:--wrap=cudaMemcpyAsync"
    "LINKER:--wrap=cudaStreamSynchronize")
  target_compile_definitions(ninfer_artifact_materialization_test PRIVATE
    NINFER_HAS_CUDA_FAULT_WRAP=1)
endif()

add_test(NAME ninfer_artifact_writer_interop_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/writer_interop.py"
    $<TARGET_FILE:ninfer_artifact_materialization_test>)

set_tests_properties(
  ninfer_artifact_writer_interop_test
  PROPERTIES SKIP_RETURN_CODE 77)

set_tests_properties(
  ninfer_artifact_materialization_test
  PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME ninfer_artifact_cpu_materialization_test
  COMMAND ninfer_artifact_materialization_test --cpu-only)
set_tests_properties(ninfer_artifact_cpu_materialization_test
  PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=")
