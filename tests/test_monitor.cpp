#include "serve/observations.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::serve;

namespace {
RequestLogContext context(std::uint64_t id, std::string http_id, bool stream) {
    RequestLogContext value;
    value.id              = id;
    value.http_request_id = std::move(http_id);
    value.protocol        = "openai";
    value.stream          = stream;
    return value;
}
} // namespace

int main() {
    int failures     = 0;
    const auto check = [&](bool pass, const char* message) {
        if (!pass) {
            std::cerr << message << '\n';
            ++failures;
        }
    };

    ServiceObservations observations;
    observations.attach();
    check(!observations.instance_id().empty(), "instance id must be assigned");
    check(observations.start_seconds() > 0.0, "start time must be a unix timestamp");

    auto subscription = observations.events.subscribe();
    check(subscription != nullptr, "first subscriber must be admitted");

    observations.preparing(7, "openai", context(7, "req-7", true));
    {
        const auto snapshot = observations.snapshot(true);
        check(snapshot.counters.submitted == 1, "preparing must count one submission");
        check(snapshot.preparing == 1, "an in-flight request must be counted as preparing");
        check(snapshot.requests.size() == 1 && snapshot.requests[0].http_request_id == "req-7",
              "the live request must be published with its HTTP id");
    }

    const auto event = subscription->next(std::chrono::milliseconds(100));
    check(event.has_value(), "preparing must publish an event");
    if (event) {
        const auto payload = nlohmann::json::parse(
            event->substr(event->find("data: ") + 6, event->size()));
        check(payload["server_instance_id"] == observations.instance_id(),
              "events must carry the instance id");
        check(payload["sequence"] == 1, "the first event must be sequence 1");
        check(payload["kind"] == "request_preparing", "the first event must be request_preparing");
        check(payload["request_id"] == 7, "events must carry the request id");
    }

    observations.first_token(7, {.prepare_seconds = 0.05, .elapsed_since_submit_seconds = 0.2});
    observations.scheduling(7, {.transition = GenerationSchedulingTransition::Paused,
                                .engine_request_id = 42});
    {
        const auto snapshot = observations.snapshot(true);
        check(snapshot.requests.size() == 1 && snapshot.requests[0].engine_request_id == 42,
              "scheduling must attach the engine request id");
        check(snapshot.counters.ttft.bins[7] == 1,
              "TTFT must observe prepare + submit-to-token at the 0.25 s bound");
    }

    GenerationOutcome outcome;
    outcome.finish_reason                            = FinishReason::OutputLimit;
    outcome.prompt_tokens                            = 11;
    outcome.completion_tokens                        = 5;
    outcome.metrics.total_seconds                    = 2.0;
    outcome.metrics.engine_timing.queue_wait_seconds = 0.1;
    observations.done(7, outcome);
    observations.rejected(8);
    observations.failed(9, true);
    observations.failed(10, false);
    observations.response_failed(7);

    const auto service = observations.snapshot(true);
    check(service.counters.submitted == 1, "rejected requests never count as submitted");
    check(service.counters.completed == 1, "a settled request must count as completed");
    check(service.counters.rejected == 1, "rejections must be counted");
    check(service.counters.cancelled == 1, "client disconnects must count as cancelled");
    check(service.counters.failed == 1, "failures must be counted");
    check(service.counters.response_failures == 1, "response failures must be counted");
    check(service.requests.empty(), "settled requests must leave the live set");

    const auto json = service_observation_json(service);
    check(json["submitted"] == 1, "service json must publish the submission count");
    check(json["preparing"] == 0, "service json must publish zero preparing once settled");
    check(json["ttft"]["bins"].size() == kRequestDurationBounds.size() + 1,
          "ttft bins must cover every bound plus the overflow bucket");
    check(json["duration_bounds_seconds"].size() == kRequestDurationBounds.size(),
          "service json must publish the histogram bounds");
    check(json["constraint"]["outcomes"].size() == 3, "constraint outcomes must have three slots");

    // A full subscriber set must be refused rather than silently dropping events.
    std::vector<std::shared_ptr<MonitorEvents::Subscription>> extra;
    for (std::size_t i = 1; i < MonitorEvents::kMaxSubscribers; ++i) {
        extra.push_back(observations.events.subscribe());
    }
    check(observations.events.subscribe() == nullptr,
          "a full subscriber set must be refused");
    for (auto& item : extra) {
        item->close();
    }

    observations.stop();
    check(subscription->closed(), "stopping the service must close subscribers");

    if (failures == 0) {
        std::cout << "monitor observations: all checks passed\n";
    }
    return failures == 0 ? 0 : 1;
}
