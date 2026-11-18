ninfer_add_test(ninfer_scheduler_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_scheduler.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_context_cost_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_context_cost.cpp"
  LIBRARIES ninfer_runtime_support ninfer::json)

ninfer_add_test(ninfer_resource_manager_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_resource_manager.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_context_capacity_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_context_capacity.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_prefix_checkpoint_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_prefix_checkpoint.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_joint_planning_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_joint_planning.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_kv_capacity_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_kv_capacity.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_sampling_defaults_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_sampling_defaults.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_grammar_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_grammar.cpp"
  LIBRARIES ninfer_grammar)

ninfer_add_test(ninfer_regex_choice_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../text/test_regex_choice.cpp"
  LIBRARIES ninfer_grammar ninfer::json)

add_test(NAME ninfer_regex_choice_oracle_test
  COMMAND ${CMAKE_COMMAND} -E env
    "NINFER_REGEX_PROBE=$<TARGET_FILE:ninfer_regex_choice_test>"
    ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tests/text/test_regex_choice.py)

ninfer_add_test(ninfer_json_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../text/test_json_schema.cpp"
  LIBRARIES ninfer_grammar ninfer::json)

add_test(NAME ninfer_json_schema_oracle_test
  COMMAND ${CMAKE_COMMAND} -E env
    "NINFER_SCHEMA_PROBE=$<TARGET_FILE:ninfer_json_schema_test>"
    ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tests/text/test_json_schema.py)

# P7.1 (V10-55): the capacity bet. Pure (core/capacity_bet.h), so no Device and no fixture: the
# test is the decision's whole contract - which bet each mode takes at a tight budget, that no bet
# claims more than the Device has, and that a bigger Device never shrinks either side.
ninfer_add_test(ninfer_capacity_bet_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_capacity_bet.cpp"
  LIBRARIES ninfer_core)

# V10-75: the host tier's ask. Pure (runtime/engine/host_arena_budget.h), so the branch that a
# healthy box cannot reach - a measured zero commit budget - is still under test, and "the budget
# could not be measured" is asserted to be a different answer from "the budget is zero".
ninfer_add_test(ninfer_host_arena_budget_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_host_arena_budget.cpp")
