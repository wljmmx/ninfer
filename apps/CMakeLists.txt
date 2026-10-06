add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
ninfer_internal_includes(ninfer)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
ninfer_internal_includes(ninfer-serve)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_logging)

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
ninfer_internal_includes(ninfer-perplexity)
target_link_libraries(ninfer-perplexity PRIVATE
  ninfer_engine ninfer_product_logging ninfer::json)

# Windows product binaries decode argv and narrow file paths as UTF-8 (the engine's whole
# text pipeline is UTF-8); the activeCodePage manifest makes the process ANSI code page
# UTF-8 system-wide on Windows 10 1903 and newer.
if(MSVC)
  foreach(app IN ITEMS ninfer ninfer-serve ninfer-perplexity)
    target_sources(${app} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/utf8-codepage.manifest")
  endforeach()
endif()
