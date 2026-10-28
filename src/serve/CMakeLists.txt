add_library(ninfer_serve STATIC
  http_transport.cpp
  request_validation.cpp
  openai_chat_request.cpp
  openai_chat_response.cpp
  openai_chat_http.cpp
  openai_common.cpp
  openai_responses_request.cpp
  openai_responses_state.cpp
  openai_responses_response.cpp
  anthropic_messages_request.cpp
  anthropic_messages_response.cpp
  anthropic_messages_http.cpp
  openai_responses_store.cpp
  translate.cpp
  generation_service.cpp
  request_events.cpp
  operational_log.cpp
  request_log.cpp
  metrics.cpp
  http_server.cpp
  openai_responses_http.cpp
  serve_options.cpp
)
ninfer_internal_includes(ninfer_serve)
target_link_libraries(ninfer_serve
  PUBLIC ninfer_engine Threads::Threads ninfer::json ninfer::httplib
  PRIVATE ninfer_media_acquire ninfer_product_logging ninfer_grammar CUDA::cudart)

# The checkpoint cache is the only thing here that needs libzmq, and it is optional: without the
# library every other feature still builds and the cache simply reports itself as unavailable.
if(TARGET ninfer::zmq)
  target_link_libraries(ninfer_serve PRIVATE ninfer::zmq)
  target_compile_definitions(ninfer_serve PRIVATE NINFER_HAVE_LMCACHE=1)
endif()
