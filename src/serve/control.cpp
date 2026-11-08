#include "serve/http_server.h"

#include "serve/control.h"

#include <chrono>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::serve {
namespace {
using Json = nlohmann::json;

Json identity(const ServiceObservations& observations) {
    return {{"server_instance_id", observations.instance_id()},
            {"timestamp_unix_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::system_clock::now().time_since_epoch())
                                      .count()}};
}

Json control_payload(const Engine::ControllerControlState& state) {
    return Json{{"ok", true}, {"controller", controller_control_json(state)}};
}

void write_control_error(httplib::Response& res, int status, std::string_view error,
                         std::string_view detail = {}) {
    Json body{{"ok", false}, {"error", std::string(error)}};
    if (!detail.empty()) { body["detail"] = std::string(detail); }
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

// A control write is acknowledged only with the state the worker actually holds. A busy engine
// parks the write and applies it at its next scheduling point; that case keeps ok=true (the
// write is real and will land) but carries detail="queued ..." so a caller can tell the two
// apart instead of reading a value nobody applied yet.
int status_for(const Engine::ControllerControlState& state) {
    return state.detail.rfind("queued", 0) == 0 ? 202 : 200;
}
} // namespace

Json controller_control_json(const Engine::ControllerControlState& state) {
    return Json{{"mode", state.mode},
                {"live", state.live},
                {"mode_live", state.mode_live},
                {"sequence", state.sequence},
                {"detail", state.detail},
                {"freeze",
                 Json{{"after", state.freeze_after},
                      {"reselect_after", state.freeze_reselect},
                      {"thaw_on_request", state.freeze_thaw_on_request},
                      {"armed", state.freeze_armed},
                      {"frozen", state.frozen},
                      {"banks", state.freeze_banks}}}};
}

void HttpServer::register_control_routes() {
    server_.Get("/control/state", [this](const httplib::Request&, httplib::Response& res) {
        if (service_ == nullptr) {
            write_control_error(res, 503, "engine is not attached yet");
            return;
        }
        try {
            auto out          = identity(observations_);
            out["ok"]         = true;
            out["controller"] = controller_control_json(service_->controller_control_state());
            res.set_header("Cache-Control", "no-store");
            res.set_content(out.dump(), "application/json");
        } catch (const std::exception& error) {
            write_control_error(res, 500, "control state unavailable", error.what());
        }
    });

    // POST /control/mode  {"mode":"auto"|"stable"|"performance"}   (also ?mode= for curl)
    server_.Post("/control/mode", [this](const httplib::Request& req, httplib::Response& res) {
        if (service_ == nullptr) {
            write_control_error(res, 503, "engine is not attached yet");
            return;
        }
        std::string mode = req.get_param_value("mode");
        if (mode.empty() && !req.body.empty()) {
            try {
                const Json body = Json::parse(req.body);
                if (!body.is_object()) {
                    write_control_error(res, 400, "body must be a JSON object");
                    return;
                }
                if (body.contains("mode")) {
                    if (!body["mode"].is_string()) {
                        write_control_error(res, 400, "\"mode\" must be a string");
                        return;
                    }
                    mode = body["mode"].get<std::string>();
                }
            } catch (const std::exception& error) {
                write_control_error(res, 400, "invalid JSON body", error.what());
                return;
            }
        }
        if (mode.empty()) {
            write_control_error(res, 400, "missing \"mode\" (auto|stable|performance)");
            return;
        }
        try {
            const auto state = service_->apply_controller_mode(mode);
            res.status       = status_for(state);
            res.set_header("Cache-Control", "no-store");
            res.set_content(control_payload(state).dump(), "application/json");
        } catch (const std::invalid_argument& error) {
            write_control_error(res, 400, "invalid mode", error.what());
        } catch (const std::exception& error) {
            write_control_error(res, 500, "mode change failed", error.what());
        }
    });

    // POST /control/freeze  {"freeze_after":N,"reselect_after":M,"thaw_on_request":bool}
    // Omitted fields keep the value in force, so a caller can move one knob without restating
    // the rest. The ranges are the bus rows' own (0..32 / 0..256 / {0,1}).
    server_.Post("/control/freeze", [this](const httplib::Request& req, httplib::Response& res) {
        if (service_ == nullptr) {
            write_control_error(res, 503, "engine is not attached yet");
            return;
        }
        Engine::ExpertFreezePolicy policy;
        try {
            const auto current = service_->controller_control_state();
            policy.freeze_after     = current.freeze_after;
            policy.reselect_after   = current.freeze_reselect;
            policy.thaw_on_request  = current.freeze_thaw_on_request;
        } catch (const std::exception& error) {
            write_control_error(res, 500, "control state unavailable", error.what());
            return;
        }
        if (req.body.empty()) {
            write_control_error(res, 400, "missing JSON body");
            return;
        }
        try {
            const Json body = Json::parse(req.body);
            if (!body.is_object()) {
                write_control_error(res, 400, "body must be a JSON object");
                return;
            }
            const auto read_u32 = [&](const char* key, std::uint32_t& out) {
                if (!body.contains(key)) { return true; }
                if (!body[key].is_number_unsigned() && !body[key].is_number_integer()) {
                    write_control_error(res, 400, std::string("\"") + key +
                                                      "\" must be a non-negative integer");
                    return false;
                }
                const auto value = body[key].get<long long>();
                if (value < 0) {
                    write_control_error(res, 400, std::string("\"") + key +
                                                      "\" must be non-negative");
                    return false;
                }
                out = static_cast<std::uint32_t>(value);
                return true;
            };
            if (!read_u32("freeze_after", policy.freeze_after) ||
                !read_u32("reselect_after", policy.reselect_after)) {
                return;
            }
            if (body.contains("thaw_on_request")) {
                if (!body["thaw_on_request"].is_boolean()) {
                    write_control_error(res, 400, "\"thaw_on_request\" must be a boolean");
                    return;
                }
                policy.thaw_on_request = body["thaw_on_request"].get<bool>();
            }
        } catch (const std::exception& error) {
            write_control_error(res, 400, "invalid JSON body", error.what());
            return;
        }
        try {
            const auto state = service_->apply_expert_freeze_policy(policy);
            res.status       = status_for(state);
            res.set_header("Cache-Control", "no-store");
            res.set_content(control_payload(state).dump(), "application/json");
        } catch (const std::invalid_argument& error) {
            write_control_error(res, 400, "invalid freeze policy", error.what());
        } catch (const std::exception& error) {
            write_control_error(res, 500, "freeze policy change failed", error.what());
        }
    });
}

} // namespace ninfer::serve
