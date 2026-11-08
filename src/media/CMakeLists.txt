add_library(ninfer_media_decode STATIC
  decode/decode.cpp
  decode/jpeg.cpp
)

# T072 (upstream 58733a86): JPEG is decoded by libjpeg-turbo so the pixels match Pillow.
# The vendored vcpkg tree keeps the package under deps/vcpkg/installed/<triplet>; honour an
# explicit -DJPEG_ROOT or CMAKE_PREFIX_PATH too.
find_package(JPEG QUIET)
if(NOT JPEG_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/../deps/vcpkg/installed/x64-windows/include/jpeglib.h")
  set(JPEG_ROOT "${PROJECT_SOURCE_DIR}/../deps/vcpkg/installed/x64-windows" CACHE PATH "libjpeg-turbo root")
  set(JPEG_INCLUDE_DIR "${JPEG_ROOT}/include")
  find_library(JPEG_LIBRARY NAMES jpeg libjpeg PATHS "${JPEG_ROOT}/lib" NO_DEFAULT_PATH)
  if(JPEG_LIBRARY)
    set(JPEG_FOUND TRUE)
    add_library(JPEG::JPEG UNKNOWN IMPORTED)
    set_target_properties(JPEG::JPEG PROPERTIES
      IMPORTED_LOCATION "${JPEG_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${JPEG_INCLUDE_DIR}")
  endif()
endif()
if(NOT JPEG_FOUND)
  message(FATAL_ERROR "libjpeg-turbo is required for the media decode JPEG path (D:/... deps/vcpkg/installed/x64-windows)")
endif()

ninfer_internal_includes(ninfer_media_decode)
target_link_libraries(ninfer_media_decode PRIVATE ${NINFER_FFMPEG_TARGET} JPEG::JPEG)
