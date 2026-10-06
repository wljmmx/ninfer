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

# Repository-pinned header dependencies. No configure-time downloads.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)

if(NINFER_BUILD_PRODUCT_SUPPORT)
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()
