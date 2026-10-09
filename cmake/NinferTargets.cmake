# Internal headers are private to each compile owner. External headers come from
# the dependency targets (or a component-local include for bundled C sources).
function(ninfer_internal_includes target)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src)
endfunction()

function(ninfer_cuda_archive target)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

function(ninfer_cuda_non_rdc_archive target)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

# Put the shared FFmpeg/CURL runtime DLLs beside an executable. copy_if_different keeps
# repeat builds cheap; NINFER_RUNTIME_DLLS is resolved once in Dependencies.cmake.
function(ninfer_deploy_runtime_dlls target)
  if(NOT WIN32 OR NOT NINFER_RUNTIME_DLLS)
    return()
  endif()
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${NINFER_RUNTIME_DLLS}
            "$<TARGET_FILE_DIR:${target}>"
    COMMENT "Deploying runtime DLLs beside ${target}"
    VERBATIM)
endfunction()
