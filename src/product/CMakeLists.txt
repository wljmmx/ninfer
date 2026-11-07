add_library(ninfer_media_acquire STATIC
  media_acquire/acquire.cpp
)
ninfer_internal_includes(ninfer_media_acquire)
target_link_libraries(ninfer_media_acquire PRIVATE ${NINFER_CURL_TARGET})
if(WIN32)
  # getaddrinfo/inet_ntop in acquire.cpp resolve through Winsock.
  target_link_libraries(ninfer_media_acquire PRIVATE ws2_32)
endif()

add_library(ninfer_product_prompt_input STATIC
  prompt_input/prompt_input.cpp
)
ninfer_internal_includes(ninfer_product_prompt_input)
target_link_libraries(ninfer_product_prompt_input PRIVATE ninfer_media_acquire ninfer::json)

add_library(ninfer_product_logging STATIC
  logging/logging.cpp
  logging/configuration.cpp
  logging/pretty_format.cpp
  logging/startup_log.cpp
)
ninfer_internal_includes(ninfer_product_logging)
target_link_libraries(ninfer_product_logging PUBLIC spdlog::spdlog)

# Unified startup configuration: one schema and one layered parser shared by both entry points.
# TOML support is the vendored toml++ (T048); the parser only reads a launch file, it never
# touches the runtime parameter bus.
add_library(ninfer_product_config STATIC
  config/options.cpp
  config/schema.cpp
  config/file.cpp
  config/configuration.cpp)
ninfer_internal_includes(ninfer_product_config)
target_link_libraries(ninfer_product_config
  PUBLIC ninfer_product_logging
  PRIVATE ninfer::toml)
