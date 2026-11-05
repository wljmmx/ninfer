#pragma once

// port (sm_89): v8 section 12, first phase - the runtime parameter bus.
//
// The environment variables are INITIAL VALUES ONLY (v8 12.1): each is read once when the bus is
// constructed, and after that the bus is the single registry of record. Re-reading getenv at
// decision time is the bug this header exists to remove; a value that lives in two places can
// disagree with itself, and a knob nobody can list is not a control surface.
//
// Every registered parameter declares the four things v8 12.1 demands before it may enter the
// bus: a hard range (lo..hi), an application cost (cost_ms), the panel signal that judges it
// (observe), and where to go when it makes things worse (rollback). A parameter without all four
// is not registered - it is described as source-missing with a collapsed [0..0] range instead,
// which is printable and honestly refuses every set().
//
// set() is REFUSAL-CAPABLE, and that is the whole difference between a controller and a pile of
// switches (v8 12.2): an out-of-range value, a caller without authority for the tier, a param in
// cooldown, or a tier whose executor has not been accepted yet (L1 pool/KV/arena resize: v8
// 13.5-5, load acceptance is another work item) all come back with a reason, and every applied
// change lands in the ledger with the throughput before and after it.
//
// Tiers (v8 12.1):
//   L0  frozen     model geometry; never writable.
//   L1  slow       expert pool / KV pages / warm arena; per 30-60s, needs reallocation; REFUSED
//                  this phase ("load acceptance pending").
//   L2  medium     concurrency / prefetch / graph / freeze family; per window (32 layers).
//   L3  fast       hot set / spill ring / second chance / per-layer counts; per layer.
//
// This phase wires exactly ONE live executor through the bus: "concurrency" (v8 9.3), attached by
// the engine's worker thread where the model code and the scheduler share one thread-local bus.
// Everything else is readable, printable and refusable - taking over more parameters is later
// phases (v8 12.5), and pretending otherwise would make the panel a work of fiction.

#include "core/controller_mode.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

enum class ParamTier : std::uint8_t {
    L0 = 0, // frozen: geometry
    L1 = 1, // slow: reallocation-class
    L2 = 2, // medium: per window
    L3 = 3, // fast: per layer
};

// set() refusals. The three the plan names (v8 12.2 note) are Range / Cooldown / NoExecutor;
// the others cover the rest of the authorization ladder.
enum class ParamReject : std::uint8_t {
    Ok = 0,
    Unknown,     // no such parameter
    TierLocked,  // L0 geometry, or a caller without authority for this tier
    LoadAccept,  // L1: the resize executors are not load-accepted yet (v8 13.5-5)
    Range,       // outside [lo..hi]
    Cooldown,    // changed within the last window
    NoExecutor,  // no accepted executor is wired for this parameter
    ApplyFailed, // the executor refused the value
};

[[nodiscard]] inline constexpr const char* param_reject_text(ParamReject reject) {
    switch (reject) {
    case ParamReject::Ok: return "ok";
    case ParamReject::Unknown: return "unknown parameter";
    case ParamReject::TierLocked: return "tier not authorized for this caller";
    case ParamReject::LoadAccept: return "load acceptance pending";
    case ParamReject::Range: return "out of range";
    case ParamReject::Cooldown: return "cooldown";
    case ParamReject::NoExecutor: return "no accepted executor";
    case ParamReject::ApplyFailed: return "executor refused";
    }
    return "unknown";
}

struct ParamSpec {
    const char* name = "";
    double initial = 0.0;  // from the environment/CLI, read once
    double lo = 0.0;       // hard bounds - the controller may not leave them
    double hi = 0.0;
    ParamTier tier = ParamTier::L2;
    double cost_ms = 0.0;  // what one application costs
    const char* observe = "";  // the panel signal that judges this parameter
    double rollback = 0.0;    // where to go when it made things worse
    const char* source = "";   // initial-value provenance, printed once per panel
    bool source_missing = false; // no readable initial-value source in the code today
};

class ParamBus {
public:
    struct LedgerEntry {
        const char* name = "";
        double from = 0.0;
        double to = 0.0;
        double tps_before = -1.0;
        double tps_after = -1.0; // < 0 until the next window settles it
        std::uint64_t layer = 0;
    };

    ParamBus() { register_startup_params(); }

    // The one write path. `from_controller` is the authority v8 12.1 gives the controller over
    // L2/L3; anyone else asking is refused with TierLocked rather than silently trusted.
    ParamReject set(const char* name, double value, bool from_controller = false) {
        const int index = find(name);
        if (index < 0) { return ParamReject::Unknown; }
        ParamSpec& spec = specs_[static_cast<std::size_t>(index)];
        if (spec.tier == ParamTier::L0) { return ParamReject::TierLocked; }
        if (spec.tier == ParamTier::L1) {
            // v8 13.5-5: Engine::resize_expert_pool/resize_kv_pages exist but are not
            // load-accepted under traffic, and no controller-side caller has been accepted
            // either. L1 stays read-only until that work item closes.
            return ParamReject::LoadAccept;
        }
        if (!from_controller) { return ParamReject::TierLocked; }
        if (std::isnan(value) || value < spec.lo || value > spec.hi) {
            return ParamReject::Range;
        }
        if (values_[static_cast<std::size_t>(index)] == value) { return ParamReject::Ok; }
        if (cooldowns_[static_cast<std::size_t>(index)] > 0) { return ParamReject::Cooldown; }
        auto& applier = appliers_[static_cast<std::size_t>(index)];
        if (!applier) { return ParamReject::NoExecutor; }
        if (!applier(value)) { return ParamReject::ApplyFailed; }
        values_[static_cast<std::size_t>(index)]      = value;
        cooldowns_[static_cast<std::size_t>(index)]   = kCooldownLayers;
        last_change_layer_[static_cast<std::size_t>(index)] = layers_;
        return ParamReject::Ok;
    }

    [[nodiscard]] double get(const char* name) const {
        const int index = find(name);
        return index < 0 ? std::nan("") : values_[static_cast<std::size_t>(index)];
    }

    // v8 12.2 accounting: (name, from, to, tps_before, tps_after), printable. The bus never
    // invents a throughput - the caller supplies what it measured, and a tps_after < 0 means
    // "not settled yet"; settle_pending() fills it once the next window has a number.
    void record(const char* name, double from, double to, double tps_before, double tps_after) {
        ledger_.push_back(LedgerEntry{name, from, to, tps_before, tps_after, layers_});
        if (ledger_.size() > kLedgerCapacity) { ledger_.erase(ledger_.begin()); }
    }

    void settle_pending(double tps) {
        if (tps <= 0.0) { return; }
        for (auto it = ledger_.rbegin(); it != ledger_.rend(); ++it) {
            if (it->tps_after >= 0.0) { break; } // only the newest open entry
            it->tps_after = tps;
            break;
        }
    }

    // The owner of a live mechanism publishes its state here (e.g. the prefetch actuator's
    // queue switch, the freeze rule's armed flag). This is a mirror, not a write path: it
    // bypasses nothing because it installs no value anywhere - the mechanism already holds it.
    void note_value(const char* name, double value) {
        const int index = find(name);
        if (index >= 0) { values_[static_cast<std::size_t>(index)] = value; }
    }

    [[nodiscard]] bool has_applier(const char* name) const {
        const int index = find(name);
        return index >= 0 && static_cast<bool>(appliers_[static_cast<std::size_t>(index)]);
    }

    [[nodiscard]] double rollback_of(const char* name) const {
        const int index = find(name);
        return index < 0 ? std::nan("") : specs_[static_cast<std::size_t>(index)].rollback;
    }

    // Installs the accepted executor for a parameter and pins its real range/initial value.
    // Used by the engine worker thread for "concurrency" (v8 9.3): the bus is thread-local and
    // the engine worker is the thread that both runs the model (the controller) and schedules
    // lanes, so the registration is race-free by construction.
    void attach_applier(const char* name, double initial, double lo, double hi,
                        std::function<bool(double)> applier) {
        const int index = find(name);
        if (index < 0) { return; }
        ParamSpec& spec           = specs_[static_cast<std::size_t>(index)];
        spec.initial              = initial;
        spec.lo                   = lo;
        spec.hi                   = hi;
        spec.rollback             = initial; // rise back = the configured maximum
        values_[static_cast<std::size_t>(index)]    = initial;
        appliers_[static_cast<std::size_t>(index)]  = std::move(applier);
        cooldowns_[static_cast<std::size_t>(index)] = 0;
    }

    // One layer tick: call before the controller's step() in the same layer so a change made at
    // layer N is re-settable at the earliest at N+32 (one decision window, v8 12.4 step 4/5).
    void tick() {
        ++layers_;
        for (auto& cooldown : cooldowns_) {
            if (cooldown > 0) { --cooldown; }
        }
    }

    // [bus] panel, same cadence as [ctl] (every 40 layers) and behind the same log switches -
    // the caller decides when printing is on, the bus decides when it is due.
    void report(bool force = false) {
        if (!force && layers_ % kReportLayers != 0) { return; }
        std::fprintf(stderr, "[bus] v8-12.3 params=%zu mode=%s layers=%llu\n", specs_.size(),
                     ninfer::controller_mode_name(ninfer::controller_mode()),
                     static_cast<unsigned long long>(layers_));
        for (int tier = 0; tier <= 3; ++tier) {
            if (tier == static_cast<int>(ParamTier::L0) && !has_tier(ParamTier::L0)) { continue; }
            print_tier(static_cast<ParamTier>(tier));
        }
        if (!ledger_.empty()) {
            std::fprintf(stderr, "[bus] ledger n=%zu latest:", ledger_.size());
            const std::size_t first = ledger_.size() > 3 ? ledger_.size() - 3 : 0;
            for (std::size_t i = first; i < ledger_.size(); ++i) {
                const auto& entry = ledger_[i];
                std::fprintf(stderr, " %s %g->%g @L%llu tps %.1f->%s", entry.name, entry.from,
                             entry.to, static_cast<unsigned long long>(entry.layer),
                             entry.tps_before,
                             entry.tps_after >= 0.0
                                 ? std::to_string(entry.tps_after).c_str()
                                 : "n/a");
            }
            std::fprintf(stderr, "\n");
        }
        std::fflush(stderr);
    }

    [[nodiscard]] std::uint64_t layers() const { return layers_; }

    // Read-only ledger view: the accounting the panel prints, for tests and future tooling.
    [[nodiscard]] const std::vector<LedgerEntry>& ledger() const { return ledger_; }

    // Test/future seam: registers a spec beyond the v8-12.3 startup table (an L0 row for the
    // tier-lock test, or a parameter a later phase adds). Production code registers everything
    // in the constructor; nothing outside tests calls this today.
    void register_spec(ParamSpec spec) {
        specs_.push_back(spec);
        values_.push_back(spec.initial);
        appliers_.emplace_back();
        cooldowns_.push_back(0);
        last_change_layer_.push_back(0);
    }

private:
    static constexpr std::uint32_t kCooldownLayers = 32; // one decision window between changes
    static constexpr std::uint64_t kReportLayers = 40;  // matches the [ctl] panel cadence
    static constexpr std::size_t kLedgerCapacity = 64;

    // Environment is read ONCE, here, at construction time. No static caching inside these
    // helpers: a second bus built by a test after changing the environment must see the change.
    static double env_number(const char* name, double fallback) {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0') { return fallback; }
        return std::strtod(value, nullptr);
    }
    static double env_presence(const char* name) {
        return std::getenv(name) != nullptr ? 1.0 : 0.0;
    }
    static double env_switch(const char* name) { // set and not "0"
        const char* value = std::getenv(name);
        return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0 ? 1.0 : 0.0;
    }

    void register_startup_params() {
        // ---- L1: reallocation-class, refused this phase (load acceptance pending) ----
        register_spec({.name = "expert_pool_mb",
                       .initial = env_number("NINFER_EXPERT_CACHE_MB", -1.0),
                       .lo = 1024.0,
                       .hi = 19456.0,
                       .tier = ParamTier::L1,
                       .cost_ms = 3000.0,
                       .observe = "coverage/miss/tps",
                       .rollback = env_number("NINFER_EXPERT_CACHE_MB", -1.0),
                       .source = "NINFER_EXPERT_CACHE_MB"});
        register_spec({.name = "kv_pages",
                       .initial = -1.0,
                       .lo = 0.0,
                       .hi = 0.0,
                       .tier = ParamTier::L1,
                       .cost_ms = 3000.0,
                       .observe = "kv-hit/evict",
                       .rollback = 0.0,
                       .source = "missing: --kv-capacity is a serve CLI value; the models "
                                 "layer cannot read it and L1 is refused this phase anyway",
                       .source_missing = true});
        register_spec({.name = "warm_arena_mb",
                       .initial = env_number("NINFER_BUDGET_RAM_MB", -1.0),
                       .lo = 256.0,
                       .hi = 8192.0,
                       .tier = ParamTier::L1,
                       .cost_ms = 3000.0,
                       .observe = "ledger.ram/cold-read",
                       .rollback = env_number("NINFER_BUDGET_RAM_MB", -1.0),
                       .source = "NINFER_BUDGET_RAM_MB"});

        // ---- L2: per-window. Only "concurrency" has an accepted executor this phase. ----
        register_spec({.name = "concurrency",
                       .initial = -1.0, // pinned by the engine worker thread at attach
                       .lo = 0.0,
                       .hi = 0.0,
                       .tier = ParamTier::L2,
                       .cost_ms = 0.0,
                       .observe = "static/tps",
                       .rollback = 0.0,
                       .source = "--max-concurrency via engine attach"});
        register_spec({.name = "lanes",
                       .initial = -1.0, // mirrored from the engine at concurrency attach
                       .lo = 1.0,
                       .hi = 8.0,
                       .tier = ParamTier::L2,
                       .cost_ms = 0.0,
                       .observe = "[st]/ens",
                       .rollback = 0.0,
                       .source = "derived from --max-concurrency (in-batch lane ceiling)"});
        register_spec({.name = "prefetch_on",
                       .initial = env_presence("NINFER_PREFETCH"),
                       .lo = 0.0,
                       .hi = 1.0,
                       .tier = ParamTier::L2,
                       .cost_ms = 0.0,
                       .observe = "useful/tps",
                       .rollback = env_presence("NINFER_PREFETCH"),
                       .source = "NINFER_PREFETCH"});
        register_spec({.name = "prefetch_slots",
                       .initial = env_number("NINFER_PREFETCH_SLOTS", 0.0),
                       .lo = 0.0,
                       .hi = 256.0,
                       .tier = ParamTier::L2,
                       .cost_ms = 0.0, // re-slicing an arena segment is milliseconds
                       .observe = "useful",
                       .rollback = env_number("NINFER_PREFETCH_SLOTS", 0.0),
                       .source = "NINFER_PREFETCH_SLOTS"});
        register_spec({.name = "prefetch_mode",
                       .initial = [] {
                           const char* value = std::getenv("NINFER_PREFETCH_MODE");
                           if (value == nullptr || *value == '\0') { return 0.0; }
                           if (std::strcmp(value, "warm") == 0) { return 1.0; }
                           if (std::strcmp(value, "vram") == 0) { return 2.0; }
                           return std::strtod(value, nullptr); // numeric form; 0 = fill
                       }(),
                       .lo = 0.0,
                       .hi = 2.0,
                       .tier = ParamTier::L2,
                       .cost_ms = 0.0,
                       .observe = "useful",
                       .rollback = 0.0,
                       .source = "NINFER_PREFETCH_MODE (0=fill 1=warm 2=vram)"});
        register_spec({.name = "graph_on",
                       .initial = [] {
                           const char* value = std::getenv("NINFER_LAYER_GRAPH");
                           return value != nullptr && std::strcmp(value, "0") == 0 ? 0.0 : 1.0;
                       }(), // default on; NINFER_LAYER_GRAPH=0 is the hard off
                       .lo = 0.0,
                       .hi = 1.0,
                       .tier = ParamTier::L2,
                       .cost_ms = 0.0, // the capture cost is paid once, not per flip
                       .observe = "tps/ens",
                       .rollback = 1.0,
                       .source = "NINFER_LAYER_GRAPH"});
        register_spec({.name = "freeze_on",
                       .initial = std::max(env_switch("NINFER_FREEZE_RULE"),
                                           env_presence("NINFER_FREEZE_ON_NO_MISS")),
                       .lo = 0.0,
                       .hi = 1.0,
                       .tier = ParamTier::L2,
                       .cost_ms = 0.0,
                       .observe = "static/tps",
                       .rollback = 0.0,
                       .source = "NINFER_FREEZE_RULE / NINFER_FREEZE_ON_NO_MISS (switch)"});
        register_spec({.name = "lmcache_restore",
                       .initial = env_switch("NINFER_LMCACHE_RESTORE"),
                       .lo = 0.0,
                       .hi = 1.0,
                       .tier = ParamTier::L2,
                       .cost_ms = 0.0,
                       .observe = "hit-rate/TTFT",
                       .rollback = 0.0,
                       .source = "NINFER_LMCACHE_RESTORE"});
        register_spec({.name = "lmcache_async",
                       .initial = env_presence("NINFER_LMCACHE_ASYNC_WRITE"),
                       .lo = 0.0,
                       .hi = 1.0,
                       .tier = ParamTier::L2,
                       .cost_ms = 0.0,
                       .observe = "write-queue-depth",
                       .rollback = 0.0,
                       .source = "NINFER_LMCACHE_ASYNC_WRITE"});
        register_spec({.name = "host_direct",
                       .initial = env_presence("NINFER_HOST_DIRECT"),
                       .lo = 0.0,
                       .hi = 1.0,
                       .tier = ParamTier::L2,
                       .cost_ms = 0.0,
                       .observe = "[route] host/cpu",
                       .rollback = 0.0,
                       .source = "NINFER_HOST_DIRECT"});
        register_spec({.name = "freeze_after",
                       .initial = env_number("NINFER_EXPERT_FREEZE", 4.0),
                       .lo = 0.0,
                       .hi = 32.0, // 0 = never freeze
                       .tier = ParamTier::L2,
                       .cost_ms = 0.0, // rewrites the S stripe + device set once
                       .observe = "[freeze] state/covers_all",
                       .rollback = 0.0,
                       .source = "NINFER_EXPERT_FREEZE"});
        register_spec({.name = "freeze_reselect",
                       .initial = env_number("NINFER_FREEZE_REFRESH", 0.0),
                       .lo = 0.0,
                       .hi = 256.0, // 0 = never re-select
                       .tier = ParamTier::L2,
                       .cost_ms = 0.0, // the table moves during a reselect window (no graphs)
                       .observe = "[freeze] reselect/tps",
                       .rollback = 0.0,
                       .source = "NINFER_FREEZE_REFRESH"});
        register_spec({.name = "freeze_thaw_on_request",
                       .initial = env_switch("NINFER_FREEZE_THAW_REQUEST"),
                       .lo = 0.0,
                       .hi = 1.0,
                       .tier = ParamTier::L2,
                       .cost_ms = 0.0, // one thaw + freeze_after calls of warm-up
                       .observe = "inter-request-tps/[freeze] thawed",
                       .rollback = 0.0,
                       .source = "NINFER_FREEZE_THAW_REQUEST"});

        // ---- L3: per-layer, microseconds. No executors this phase. ----
        register_spec({.name = "hot_cap",
                       .initial = env_number("NINFER_EXPERT_EPOCH", 0.0),
                       .lo = 8.0,
                       .hi = 10240.0, // the full routed expert count
                       .tier = ParamTier::L3,
                       .cost_ms = 0.001,
                       .observe = "coverage/ens",
                       .rollback = env_number("NINFER_EXPERT_EPOCH", 0.0),
                       .source = "NINFER_EXPERT_EPOCH (epoch-static hot region)"});
        register_spec({.name = "spill_slots",
                       .initial = env_number("NINFER_EXPERT_RING_SLOTS", 8.0),
                       .lo = 8.0,
                       .hi = 3413.0, // cap/3 at the full 10240-expert pool
                       .tier = ParamTier::L3,
                       .cost_ms = 0.001,
                       .observe = "ring-hit",
                       .rollback = 8.0, // B4: 8 is the measured p99 everywhere
                       .source = "auto min(8*lanes, cap/3); NINFER_EXPERT_RING_SLOTS overrides"});
        register_spec({.name = "second_chance_slots",
                       .initial = std::getenv("NINFER_WARM_SECOND_CHANCE_SLOTS") != nullptr
                                      ? env_number("NINFER_WARM_SECOND_CHANCE_SLOTS", 0.0)
                                      : env_presence("NINFER_HOST_FREEZE") * 160.0,
                       .lo = 0.0,
                       .hi = 248.0, // routed/2 at the default 496-slot arena
                       .tier = ParamTier::L3,
                       .cost_ms = 0.001,
                       .observe = "2nd-chance-hit/ens",
                       .rollback = 0.0, // Q53: no positive yield measured; the default is 0
                       .source = "NINFER_WARM_SECOND_CHANCE_SLOTS"});
        register_spec({.name = "admit_min",
                       .initial = 0.0,
                       .lo = 0.0,
                       .hi = 1.0,
                       .tier = ParamTier::L3,
                       .cost_ms = 0.001,
                       .observe = "2nd-chance-hit-rate",
                       .rollback = 0.0,
                       .source = "missing: no admission-score floor exists in the code yet "
                                 "(v8 12.3 lists it as to-be-added)",
                       .source_missing = true});
        register_spec({.name = "per_layer",
                       .initial = env_number("NINFER_PREFETCH_PER_LAYER", 8.0),
                       .lo = 0.0,
                       .hi = 8.0,
                       .tier = ParamTier::L3,
                       .cost_ms = 0.001,
                       .observe = "useful",
                       .rollback = env_number("NINFER_PREFETCH_PER_LAYER", 8.0),
                       .source = "NINFER_PREFETCH_PER_LAYER"});
        register_spec({.name = "warm_static_slots",
                       .initial = env_number("NINFER_WARM_STATIC_SLOTS", 0.0),
                       .lo = 0.0,
                       .hi = 256.0, // runtime clamps to routed/bank-1 as well
                       .tier = ParamTier::L3,
                       .cost_ms = 0.0, // re-splitting S/D and refilling S is milliseconds
                       .observe = "[freeze] host_static/present,capt_ready",
                       .rollback = 0.0, // 0 = the complement set
                       .source = "NINFER_WARM_STATIC_SLOTS"});
    }

    [[nodiscard]] int find(const char* name) const {
        for (std::size_t i = 0; i < specs_.size(); ++i) {
            if (std::strcmp(specs_[i].name, name) == 0) { return static_cast<int>(i); }
        }
        return -1;
    }

    [[nodiscard]] bool has_tier(ParamTier tier) const {
        return std::any_of(specs_.begin(), specs_.end(),
                           [tier](const ParamSpec& spec) { return spec.tier == tier; });
    }

    void print_tier(ParamTier tier) const {
        const char* label = tier == ParamTier::L1 ? "L1 ro(load acceptance pending)"
                          : tier == ParamTier::L2 ? "L2"
                          : tier == ParamTier::L3 ? "L3"
                                                 : "L0 frozen";
        std::fprintf(stderr, "[bus] %s:", label);
        for (std::size_t i = 0; i < specs_.size(); ++i) {
            const ParamSpec& spec = specs_[i];
            if (spec.tier != tier) { continue; }
            const double value = values_[i];
            std::fprintf(stderr, " %s=", spec.name);
            if (value < 0.0) {
                std::fprintf(stderr, "n/a");
            } else {
                std::fprintf(stderr, "%g", value);
            }
            std::fprintf(stderr, "[%g..%g] obs=%s", spec.lo, spec.hi, spec.observe);
            if (spec.source_missing) { std::fprintf(stderr, " src-missing"); }
            if (tier == ParamTier::L1) {
                // every L1 write is refused this phase; the state is the refusal reason
            } else if (appliers_[i]) {
                std::fprintf(stderr, " live");
            } else if (cooldowns_[i] > 0) {
                std::fprintf(stderr, " no-executor cooling(%u)", cooldowns_[i]);
            } else {
                std::fprintf(stderr, " no-executor");
            }
            std::fprintf(stderr, " (%s)", spec.source);
        }
        std::fprintf(stderr, "\n");
    }

    std::vector<ParamSpec> specs_;
    std::vector<double> values_;
    std::vector<std::function<bool(double)>> appliers_;
    std::vector<std::uint32_t> cooldowns_;
    std::vector<std::uint64_t> last_change_layer_;
    std::vector<LedgerEntry> ledger_;
    std::uint64_t layers_ = 0;
};

// The bus is thread-local on purpose: the engine worker thread is the thread that runs the model
// (whose controller decides) and the scheduler (whose lane count is the executor), so one
// thread-local bus is exactly the shared state those two need - and no other thread can reach it.
inline ParamBus& param_bus() {
    static thread_local ParamBus bus;
    return bus;
}

} // namespace ninfer::models::qwen3_5::execution
