find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

if(WIN32)
  # Windows builds resolve FFmpeg through vcpkg in manifest mode (see vcpkg.json) or
  # through an externally provided FFmpeg build: pass FFMPEG_INCLUDE_DIRS,
  # FFMPEG_LIBRARY_DIRS and FFMPEG_LIBRARIES as cache variables and the find_module
  # lookup is skipped. Any shared FFmpeg with headers and MSVC import libraries works.
  if(NOT (FFMPEG_INCLUDE_DIRS AND FFMPEG_LIBRARY_DIRS AND FFMPEG_LIBRARIES))
    find_package(FFMPEG REQUIRED)
  endif()
  add_library(ninfer_ffmpeg_dependencies INTERFACE)
  target_include_directories(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_INCLUDE_DIRS})
  target_link_directories(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARY_DIRS})
  target_link_libraries(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARIES})
  set(NINFER_FFMPEG_TARGET ninfer_ffmpeg_dependencies)

  if(NINFER_BUILD_PRODUCT_SUPPORT)
    # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
    # introduced in libcurl 7.85 (not merely the version of the maintainer environment).
    find_package(CURL 7.85 REQUIRED)
    set(NINFER_CURL_TARGET CURL::libcurl)
  endif()
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libswscale)
  set(NINFER_FFMPEG_TARGET PkgConfig::FFMPEG)

  if(NINFER_BUILD_PRODUCT_SUPPORT)
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
    set(NINFER_CURL_TARGET PkgConfig::LIBCURL)
  endif()
endif()

# Windows executables import FFmpeg and CURL through their shared libraries, so the
# runtime DLLs must sit beside every binary. The import libraries alone are not enough:
# a process whose DLL cannot be resolved never starts, and under CTest the loader's
# error dialog blocks, which makes an offending test look like a hang. Resolve the DLL
# directories once here; every test/bench/app target deploys them (see
# ninfer_deploy_runtime_dlls in cmake/NinferTargets.cmake).
set(NINFER_RUNTIME_DLLS "")
set(FFMPEG_RUNTIME_DIR "" CACHE PATH "Directory holding the FFmpeg runtime DLLs")
set(CURL_RUNTIME_DIR "" CACHE PATH "Directory holding the CURL runtime DLLs")
if(WIN32)
  if(NOT FFMPEG_RUNTIME_DIR)
    # The externally provided FFmpeg layout keeps import libraries in lib/ and the DLLs
    # in the sibling bin/.
    foreach(lib_dir IN LISTS FFMPEG_LIBRARY_DIRS)
      if(EXISTS "${lib_dir}/../bin")
        get_filename_component(FFMPEG_RUNTIME_DIR "${lib_dir}/../bin" ABSOLUTE)
        break()
      endif()
    endforeach()
  endif()
  if(FFMPEG_RUNTIME_DIR AND EXISTS "${FFMPEG_RUNTIME_DIR}")
    file(GLOB NINFER_FFMPEG_RUNTIME_DLLS CONFIGURE_DEPENDS "${FFMPEG_RUNTIME_DIR}/*.dll")
    list(APPEND NINFER_RUNTIME_DLLS ${NINFER_FFMPEG_RUNTIME_DLLS})
  endif()

  # find_package(CURL) reports <prefix>/lib/cmake/CURL; the DLL lives in <prefix>/bin.
  if(NOT CURL_RUNTIME_DIR AND CURL_DIR)
    get_filename_component(NINFER_CURL_PREFIX "${CURL_DIR}/../../.." ABSOLUTE)
    if(EXISTS "${NINFER_CURL_PREFIX}/bin")
      set(CURL_RUNTIME_DIR "${NINFER_CURL_PREFIX}/bin")
    endif()
  endif()
  if(CURL_RUNTIME_DIR AND EXISTS "${CURL_RUNTIME_DIR}")
    file(GLOB NINFER_CURL_RUNTIME_DLLS CONFIGURE_DEPENDS "${CURL_RUNTIME_DIR}/libcurl*.dll")
    list(APPEND NINFER_RUNTIME_DLLS ${NINFER_CURL_RUNTIME_DLLS})
  endif()

  list(REMOVE_DUPLICATES NINFER_RUNTIME_DLLS)
  if(NINFER_RUNTIME_DLLS)
    list(LENGTH NINFER_RUNTIME_DLLS NINFER_RUNTIME_DLL_COUNT)
    message(STATUS "NInfer: deploying ${NINFER_RUNTIME_DLL_COUNT} runtime DLL(s) beside each executable")
  elseif(NINFER_BUILD_PRODUCT_SUPPORT)
    message(WARNING
      "NInfer: shared FFmpeg/CURL were linked but no runtime DLL directory was found; "
      "set FFMPEG_RUNTIME_DIR / CURL_RUNTIME_DIR or the product binaries will not start")
  endif()
endif()

# Repository-pinned header dependencies. No configure-time downloads.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)

# CPU grammar source base; consumers will link it explicitly.
add_subdirectory(third_party/xgrammar EXCLUDE_FROM_ALL)

if(NINFER_BUILD_PRODUCT_SUPPORT)
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()
