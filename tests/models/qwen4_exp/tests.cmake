ninfer_add_test(ninfer_ple_disk_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ple_disk.cpp"
  LIBRARIES ninfer_ple ninfer::json)
