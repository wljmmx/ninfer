#include "product/config/options.h"
#include "product/logging/configuration.h"
#include "product/logging/logging.h"
#include "product/logging/startup_log.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"

#include <spdlog/logger.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace {

// The handler only raises a flag: everything it could reach from a signal context is a lock or a
// socket operation, and neither belongs there. A monitor thread performs the stop, which keeps the
// ordering (engine first, then the HTTP worker join) in one place.
static_assert(std::atomic<bool>::is_always_lock_free);
std::atomic<bool> g_stop_requested{false};

void handle_signal(int) { g_stop_requested.store(true, std::memory_order_relaxed); }

} // namespace

int main(int argc, char** argv) {
    ninfer::product::ServeOptions options;
    try {
        options = ninfer::product::parse_serve_options(argc, argv);
    } catch (const std::invalid_argument& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        std::cerr << ninfer::product::startup_usage_text(argv[0], ninfer::product::EntryPoint::Serve);
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        return 1;
    }
    if (options.help_requested) {
        std::cout << ninfer::product::startup_usage_text(argv[0], ninfer::product::EntryPoint::Serve);
        return 0;
    }

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer-serve",
         .level        = options.launch.logging.level,
         .presentation = ninfer::product::LogPresentation::Service});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);
    ninfer::serve::OperationalLog operational_log(logger);
    bool serving = false;

    try {
        ninfer::product::log_configuration(*logger, ninfer::product::configuration_view(options));
        ninfer::serve::HttpServer server(options, logger);
        if (!server.bind()) {
            operational_log.bind_failure(options.host, options.port);
            return 1;
        }

        ninfer::serve::GenerationService service(options, startup_log.observer());
        startup_log.engine_ready(service.load_summary());
        operational_log.engine_capacity(service);

        // Printed rather than assumed: these are the numbers an external cache is registered with,
        // and they come from the live page layout and state pool. A zero means a pool is not up.
        {
            const ninfer::CacheGeometry geometry = service.cache_geometry();
            logger->info("cache geometry: page {} B, {} token(s) per page, state image {} B",
                         geometry.page_bytes, geometry.page_tokens,
                         geometry.state_image_bytes);
        }
        // Off unless NINFER_LMCACHE_RESTORE asks for it, and never fatal: the cache is an
        // accelerator, so a server without it serves every request the same way.
        logger->info("checkpoint cache | {}",
                     service.attach_checkpoint_cache() ? "attached" : "off");

        using Clock                            = std::chrono::steady_clock;
        const Clock::time_point warmup_started = Clock::now();
        operational_log.warmup_started();
        try {
            service.warmup();
        } catch (const std::exception& exception) {
            const double seconds =
                std::chrono::duration<double>(Clock::now() - warmup_started).count();
            operational_log.warmup_failure(seconds, exception.what());
            return 1;
        }
        operational_log.warmup_complete(
            std::chrono::duration<double>(Clock::now() - warmup_started).count());
        server.attach(service);

        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);
#ifdef SIGBREAK
        // Windows has no SIGTERM: CTRL_BREAK_EVENT is the console equivalent, and the runtime maps it
        // to SIGBREAK. Without it a Windows server can only be killed, never drained.
        std::signal(SIGBREAK, handle_signal);
#endif

        serving = true;
        operational_log.server_ready(options.host, options.port, server.public_model_id(),
                                     !options.api_key.empty());

        // 10 ms is the cost of a shutdown request waiting behind a sleep, and this is one thread
        // that does nothing else; a condition variable would make the handler take a lock.
        std::jthread stop_monitor([&server](std::stop_token stop) {
            while (!stop.stop_requested()) {
                if (g_stop_requested.load(std::memory_order_relaxed)) {
                    server.stop();
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
        const bool ok = server.listen();
        stop_monitor.request_stop();
        if (!ok) {
            operational_log.listen_failure(options.host, options.port);
            return 1;
        }
        operational_log.server_stopped();
        return 0;
    } catch (const std::exception& exception) {
        operational_log.server_failure(serving, exception.what());
        return 1;
    }
}
