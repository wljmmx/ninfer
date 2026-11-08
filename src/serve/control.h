#pragma once

// T131: the control plane's HTTP face. One JSON shape for the engine's controller state, used by
// both /monitor/info (the read-only dashboard panel) and /control/* (the writes), so the console
// never has to reconcile two spellings of the same three knobs.

#include "ninfer/engine.h"

#include <nlohmann/json.hpp>

namespace ninfer::serve {

[[nodiscard]] nlohmann::json
controller_control_json(const Engine::ControllerControlState& state);

} // namespace ninfer::serve
