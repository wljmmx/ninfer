// port (route B / P1): Win32 implementation of the CPU topology, affinity and feature API that
// upstream implements with hwloc + <cpuid.h>. Same header, same semantics; upstream's Linux file is
// left untouched so merges stay mechanical. The Windows side uses:
//   * hwloc discovery      -> GetLogicalProcessorInformationEx (RelationProcessorCore/RelationCache)
//   * hwloc allowed cpuset -> GetThreadGroupAffinity  (per-thread, like sched_getaffinity(0);
//                             the process mask is a superset and would hide a bind)
//   * hwloc_set_cpubind    -> SetThreadGroupAffinity
//   * __get_cpuid/xgetbv   -> MSVC __cpuid/__cpuidex + _xgetbv
// The shared parts (worker startup, feature decoding, ordering helpers) are intentionally identical
// to the Linux implementation so the two cannot drift.
#include "core/cpu/system.h"

#include <windows.h>

#include <intrin.h>

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <set>
#include <stdexcept>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

namespace ninfer {
namespace {

[[noreturn]] void fail(const char* operation) {
    throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), operation);
}

// The calling thread's CPU set, the Windows analogue of sched_getaffinity(0). A thread's affinity
// is one processor group plus a mask, so global CPU ids are group * 64 + bit.
CpuAffinity thread_affinity() {
    GROUP_AFFINITY affinity{};
    if (!GetThreadGroupAffinity(GetCurrentThread(), &affinity)) fail("read thread affinity");
    CpuAffinity result;
    for (unsigned bit = 0; bit < sizeof(KAFFINITY) * 8; ++bit)
        if (affinity.Mask & (static_cast<KAFFINITY>(1) << bit))
            result.push_back(static_cast<std::uint32_t>(affinity.Group) * 64U + bit);
    return result;
}

} // namespace

struct CpuTopology::Impl {
    CpuTopologyInfo info;
};

CpuTopology::CpuTopology() : impl_(std::make_unique<Impl>()) {
    auto& info         = impl_->info;
    const auto allowed = thread_affinity();
    if (allowed.empty()) throw std::runtime_error("no available CPUs");

    DWORD length = 0;
    (void)GetLogicalProcessorInformationEx(RelationCache, nullptr, &length);
    if (!length) throw std::runtime_error("CPU cache discovery is unavailable");
    std::vector<std::byte> buffer(length);
    if (!GetLogicalProcessorInformationEx(
            RelationCache,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &length))
        fail("read CPU cache information");
    for (std::size_t offset = 0; offset < length;) {
        const auto* entry =
            reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
        if (entry->Relationship == RelationCache && entry->Cache.Type != CacheUnified &&
            entry->Cache.Type != CacheTrace) {
            info.caches.push_back({static_cast<std::uint32_t>(info.caches.size()),
                                   static_cast<std::uint32_t>(entry->Cache.Level),
                                   static_cast<std::uint64_t>(entry->Cache.CacheSize),
                                   static_cast<std::uint32_t>(entry->Cache.LineSize)});
        }
        offset += entry->Size;
    }

    length = 0;
    (void)GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    if (!length) throw std::runtime_error("CPU core discovery is unavailable");
    buffer.assign(length, std::byte{0});
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &length))
        fail("read CPU core information");
    std::uint32_t core_index = 0;
    std::set<std::uint32_t> seen;
    std::uint32_t deepest    = 0;
    for (std::uint32_t index = 1; index < info.caches.size(); ++index)
        if (info.caches[index].level > info.caches[deepest].level) deepest = index;
    for (std::size_t offset = 0; offset < length;) {
        const auto* entry =
            reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
        if (entry->Relationship == RelationProcessorCore) {
            for (WORD group_index = 0; group_index < entry->Processor.GroupCount; ++group_index) {
                const auto group    = entry->Processor.GroupMask[group_index].Group;
                const KAFFINITY mask = entry->Processor.GroupMask[group_index].Mask;
                for (unsigned bit = 0; bit < sizeof(KAFFINITY) * 8; ++bit) {
                    if (!(mask & (static_cast<KAFFINITY>(1) << bit))) continue;
                    const auto cpu = static_cast<std::uint32_t>(group) * 64U + bit;
                    if (std::find(allowed.begin(), allowed.end(), cpu) == allowed.end()) continue;
                    if (!seen.insert(cpu).second) continue;
                    CpuProcessor processor{.cpu = cpu};
                    processor.core = core_index;
                    // Like hwloc's deepest data cache: the last-level cache this processor is in.
                    if (!info.caches.empty()) processor.cache = deepest;
                    info.processors.push_back(processor);
                }
            }
            ++core_index;
        }
        offset += entry->Size;
    }
    if (info.processors.empty()) throw std::runtime_error("no available CPUs");
    std::sort(info.processors.begin(), info.processors.end(),
              [](const CpuProcessor& a, const CpuProcessor& b) { return a.cpu < b.cpu; });
}

CpuTopology::~CpuTopology() = default;

const CpuTopologyInfo& CpuTopology::info() const noexcept { return impl_->info; }

CpuAffinity current_thread_affinity(const CpuTopology&) { return thread_affinity(); }

void validate_cpu_affinity(const CpuTopologyInfo& topology, std::span<const std::uint32_t> cpus) {
    if (cpus.empty()) throw std::invalid_argument("CPU affinity must not be empty");
    std::set<std::uint32_t> seen;
    for (const auto cpu : cpus) {
        const auto found = std::find_if(topology.processors.begin(), topology.processors.end(),
                                        [=](const auto& p) { return p.cpu == cpu; });
        if (found == topology.processors.end() || !seen.insert(cpu).second)
            throw std::invalid_argument("CPU affinity is unavailable or duplicated");
    }
}

void bind_current_thread(const CpuTopology& topology, std::span<const std::uint32_t> cpus) {
    validate_cpu_affinity(topology.info(), cpus);
    if (cpus.size() > sizeof(DWORD_PTR) * 8)
        throw std::invalid_argument("CPU affinity set is too large");
    // A thread belongs to one processor group, so an affinity set may not straddle groups.
    GROUP_AFFINITY affinity{};
    affinity.Group = cpus.front() / 64U;
    for (const auto cpu : cpus) {
        if (cpu / 64U != affinity.Group)
            throw std::invalid_argument("CPU affinity must stay within one processor group");
        affinity.Mask |= static_cast<KAFFINITY>(1) << (cpu % 64U);
    }
    if (!SetThreadGroupAffinity(GetCurrentThread(), &affinity, nullptr)) fail("bind CPU thread");
}

CpuAffinity cpu_worker_order(const CpuTopologyInfo& topology, bool use_smt) {
    auto processors = topology.processors;
    std::sort(processors.begin(), processors.end(), [](const auto& a, const auto& b) {
        return std::tie(a.cache, a.core, a.cpu) < std::tie(b.cache, b.core, b.cpu);
    });
    CpuAffinity result, siblings;
    std::set<std::uint32_t> cores;
    for (const auto& p : processors) {
        if (!p.core || cores.insert(*p.core).second)
            result.push_back(p.cpu);
        else
            siblings.push_back(p.cpu);
    }
    if (use_smt) result.insert(result.end(), siblings.begin(), siblings.end());
    return result;
}

CpuTeamPlan plan_cpu_team(const CpuTopologyInfo& topology, std::uint32_t workers, bool use_smt) {
    auto cpus = cpu_worker_order(topology, use_smt);
    if (workers == 0 || workers > cpus.size())
        throw std::invalid_argument("CPU team size exceeds the selected processors");
    cpus.resize(workers);
    return {std::move(cpus)};
}

CpuFeatures cpu_detail::usable_features(const FeatureRegisters& r) noexcept {
    CpuFeatures result;
    constexpr auto avx_state = (1U << 26) | (1U << 27) | (1U << 28);
    if ((r.leaf1_ecx & avx_state) != avx_state || (r.xcr0 & 0x6U) != 0x6U) return result;
    const auto add = [&](bool available, CpuFeature feature) {
        if (available) result.bits |= static_cast<std::uint32_t>(feature);
    };
    add(r.leaf1_ecx & (1U << 12), CpuFeature::Fma);
    add(r.leaf1_ecx & (1U << 29), CpuFeature::F16c);
    add(r.leaf7_0_ebx & (1U << 5), CpuFeature::Avx2);
    add(r.leaf7_1_eax & (1U << 4), CpuFeature::AvxVnni);
    if ((r.xcr0 & 0xe6U) != 0xe6U || !(r.leaf7_0_ebx & (1U << 16))) return result;
    add(true, CpuFeature::Avx512F);
    add(r.leaf7_0_ebx & (1U << 17), CpuFeature::Avx512Dq);
    add(r.leaf7_0_ebx & (1U << 30), CpuFeature::Avx512Bw);
    add(r.leaf7_0_ebx & (1U << 31), CpuFeature::Avx512Vl);
    add(r.leaf7_0_ecx & (1U << 11), CpuFeature::Avx512Vnni);
    add(r.leaf7_1_eax & (1U << 5), CpuFeature::Avx512Bf16);
    return result;
}

CpuFeatures current_cpu_features() noexcept {
    cpu_detail::FeatureRegisters state;
    int registers[4] = {0, 0, 0, 0};
    __cpuid(registers, 1);
    state.leaf1_ecx = static_cast<std::uint32_t>(registers[2]);
    if ((state.leaf1_ecx & ((1U << 26) | (1U << 27))) == ((1U << 26) | (1U << 27)))
        state.xcr0 = _xgetbv(0);
    __cpuid(registers, 0);
    const int max_leaf = registers[0];
    if (max_leaf >= 7) {
        __cpuidex(registers, 7, 0);
        state.leaf7_0_ebx = static_cast<std::uint32_t>(registers[1]);
        state.leaf7_0_ecx = static_cast<std::uint32_t>(registers[2]);
        if (registers[0] >= 1) {
            __cpuidex(registers, 7, 1);
            state.leaf7_1_eax = static_cast<std::uint32_t>(registers[0]);
        }
    }
    return cpu_detail::usable_features(state);
}

CpuIdentity current_cpu_identity() noexcept {
    CpuIdentity result;
    int registers[4] = {0, 0, 0, 0};
    __cpuid(registers, 0);
    std::memcpy(result.vendor.data(), &registers[1], 4);
    std::memcpy(result.vendor.data() + 4, &registers[3], 4);
    std::memcpy(result.vendor.data() + 8, &registers[2], 4);
    __cpuid(registers, 1);
    const auto eax    = static_cast<std::uint32_t>(registers[0]);
    const auto ecx    = static_cast<std::uint32_t>(registers[2]);
    const auto family = (eax >> 8) & 15U;
    result.family     = family + (family == 15 ? (eax >> 20) & 255U : 0);
    result.model    = ((eax >> 4) & 15U) | ((family == 6 || family == 15) ? (eax >> 12) & 240U : 0);
    result.stepping = eax & 15U;
    result.virtualized = (ecx & (1U << 31)) != 0;
    return result;
}

std::vector<std::thread> cpu_detail::start_workers(std::uint32_t count,
                                                   std::function<void(std::uint32_t)> initialize,
                                                   WorkerBody body, void* argument) {
    if (count == 0 || !initialize || !body)
        throw std::invalid_argument("invalid CPU worker startup");

    struct Startup {
        std::mutex mutex;
        std::condition_variable changed;
        std::uint32_t ready                                   = 0;
        enum class Decision { Pending, Run, Cancel } decision = Decision::Pending;
        std::exception_ptr error;
        std::function<void(std::uint32_t)> initialize;
    };

    auto startup        = std::make_shared<Startup>();
    startup->initialize = std::move(initialize);
    std::vector<std::thread> threads;
    threads.reserve(count);
    try {
        for (std::uint32_t rank = 0; rank < count; ++rank) {
            threads.emplace_back([startup, body, argument, rank]() mutable {
                std::exception_ptr error;
                try {
                    startup->initialize(rank);
                } catch (...) { error = std::current_exception(); }
                bool run;
                {
                    std::unique_lock lock(startup->mutex);
                    if (error && !startup->error) startup->error = error;
                    ++startup->ready;
                    startup->changed.notify_all();
                    startup->changed.wait(
                        lock, [&] { return startup->decision != Startup::Decision::Pending; });
                    run = startup->decision == Startup::Decision::Run;
                }
                startup.reset();
                if (run) body(argument, rank);
            });
        }
        {
            std::unique_lock lock(startup->mutex);
            startup->changed.wait(lock, [&] { return startup->ready == count; });
            if (startup->error) std::rethrow_exception(startup->error);
            startup->decision = Startup::Decision::Run;
        }
        startup->changed.notify_all();
    } catch (...) {
        {
            std::lock_guard lock(startup->mutex);
            startup->decision = Startup::Decision::Cancel;
        }
        startup->changed.notify_all();
        join_workers(threads);
        throw;
    }
    return threads;
}

void cpu_detail::join_workers(std::span<std::thread> workers) noexcept {
    for (auto& worker : workers)
        if (worker.joinable()) worker.join();
}

} // namespace ninfer
