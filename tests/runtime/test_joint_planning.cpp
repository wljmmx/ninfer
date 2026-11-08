// T156 (upstream 049d5fb4): joint capacity planning and prefix-checkpoint retention.

#include "runtime/engine/context_cache/joint_planning.h"

#include <cstdio>
#include <vector>

namespace {

using ninfer::runtime::context_cache::JointDemand;
using ninfer::runtime::context_cache::plan_joint_capacity;
using ninfer::runtime::context_cache::PrefixCandidate;
using ninfer::runtime::context_cache::select_prefix_checkpoints;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

void test_distribution_order() {
    // 1000 total, 100 headroom, workspace 200, state 300 -> KV gets the residual 400.
    const auto budget = plan_joint_capacity(1000, JointDemand{.kv_bytes = 0,
                                                              .state_bytes = 300,
                                                              .workspace_bytes = 200},
                                            100);
    check(budget.feasible, "a fitting plan is feasible");
    check(budget.workspace_bytes == 200, "workspace gets its demand");
    check(budget.state_bytes == 300, "state gets its demand");
    check(budget.kv_bytes == 400, "KV takes the residual");
    check(budget.headroom_bytes == 100, "headroom is reported back");
}

void test_infeasible_when_mandatory_pools_do_not_fit() {
    check(!plan_joint_capacity(100, JointDemand{.workspace_bytes = 101}, 0).feasible,
          "workspace beyond the total is infeasible");
    check(!plan_joint_capacity(100, JointDemand{.state_bytes = 60, .workspace_bytes = 41}, 0)
               .feasible,
          "workspace plus state beyond the total is infeasible");
    check(plan_joint_capacity(100, JointDemand{.state_bytes = 50, .workspace_bytes = 50}, 0).feasible,
          "exactly fitting mandatory pools are feasible");
    check(plan_joint_capacity(100, JointDemand{.state_bytes = 50, .workspace_bytes = 50}, 10)
              .headroom_bytes == 10,
          "headroom is clamped inside the total");
    check(!plan_joint_capacity(10, JointDemand{.workspace_bytes = 5}, 10).feasible,
          "headroom equal to the total leaves nothing for workspace");
}

void test_prefix_selection_ranks_by_value() {
    std::vector<PrefixCandidate> candidates{
        {.handle = 1, .frontier = 100, .bytes = 100, .reused = false, .last_demand = 1},
        {.handle = 2, .frontier = 300, .bytes = 100, .reused = false, .last_demand = 5},
        {.handle = 3, .frontier = 200, .bytes = 100, .reused = true, .last_demand = 2},
        {.handle = 4, .frontier = 500, .bytes = 100, .reused = false, .last_demand = 3},
    };
    // Budget for three: reuse wins, then frontier 500, then 300; 100 is dropped.
    const auto picked = select_prefix_checkpoints(candidates, 300);
    check(picked.handles.size() == 3, "three candidates fit the budget");
    check(picked.bytes_kept == 300, "the kept bytes match the budget");
    check(picked.handles[0] == 3, "a reused candidate is ranked first");
    check(picked.handles[1] == 4, "then the longest frontier");
    check(picked.handles[2] == 2, "then the next longest");
    check(!picked.over_budget, "a fitting selection is not over budget");
}

void test_pinned_always_kept() {
    std::vector<PrefixCandidate> candidates{
        {.handle = 10, .frontier = 50, .bytes = 400, .pinned = true},
        {.handle = 11, .frontier = 900, .bytes = 400, .reused = true},
    };
    const auto picked = select_prefix_checkpoints(candidates, 500);
    check(picked.handles.size() == 1, "only the pinned candidate fits");
    check(picked.handles[0] == 10, "the pinned candidate is kept");
    check(!picked.over_budget, "pinned within budget is not over budget");

    const auto starving = select_prefix_checkpoints(candidates, 300);
    check(starving.over_budget, "pinned beyond the budget reports over budget");
    check(starving.handles.size() == 1, "the pinned set is returned for the caller to reject");
}

void test_empty_input() {
    const auto none = select_prefix_checkpoints({}, 1000);
    check(none.handles.empty() && none.bytes_kept == 0 && !none.over_budget,
          "an empty candidate set is a valid empty selection");
    const auto zero = plan_joint_capacity(0, JointDemand{}, 0);
    check(zero.feasible && zero.kv_bytes == 0, "a zero budget with zero demand fits");
}

}  // namespace

int main() {
    test_distribution_order();
    test_infeasible_when_mandatory_pools_do_not_fit();
    test_prefix_selection_ranks_by_value();
    test_pinned_always_kept();
    test_empty_input();
    if (failures == 0) { std::printf("ok\n"); }
    return failures == 0 ? 0 : 1;
}
