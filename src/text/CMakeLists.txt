add_library(ninfer_text STATIC
  unicode.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c
)
# port: utf8proc is vendored as a plain source file compiled into ninfer_text, so its header must not
# decorate the definitions with __declspec(dllimport) (MSVC: error C2491). Every consumer has to agree
# with that, or it references __imp_utf8proc_* and the executables fail to link. The header has exactly
# two consumers - src/text/unicode.cpp here and third_party/llama-jinja/jinja/unicode.cpp below - so the
# definition is scoped to those two libraries rather than applied to the whole build.
if(WIN32)
  target_compile_definitions(ninfer_text PRIVATE UTF8PROC_STATIC)
endif()

ninfer_internal_includes(ninfer_text)
target_include_directories(ninfer_text PRIVATE ${PROJECT_SOURCE_DIR}/third_party)

# Language evaluation reuses the engine's Unicode and JSON dependencies.
target_sources(ninfer_jinja PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/jinja.cpp)
ninfer_internal_includes(ninfer_jinja)
target_include_directories(ninfer_jinja PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer_jinja PRIVATE ninfer_text ninfer::json)
if(WIN32)
  target_compile_definitions(ninfer_jinja PRIVATE UTF8PROC_STATIC)
endif()

add_library(ninfer_grammar STATIC grammar.cpp json_schema.cpp schema_composition.cpp json_input.cpp)
ninfer_internal_includes(ninfer_grammar)
target_include_directories(ninfer_grammar PRIVATE ${PROJECT_SOURCE_DIR}/third_party/xgrammar/cpp)
target_include_directories(ninfer_grammar SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/xgrammar/3rdparty/picojson)
target_link_libraries(ninfer_grammar PRIVATE ninfer_xgrammar ninfer_text ninfer::json)
