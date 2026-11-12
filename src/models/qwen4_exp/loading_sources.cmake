# The family's own loading sources: its config identity and the binding vocabulary the shared text
# binder asks for through `models/load_rules.h`.
target_sources(ninfer_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load/text.cpp"
)
