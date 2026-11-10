#include "models/qwen3_5/execution/gdn.h"

#include "ninfer/ops/gdn_control.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/w4a8.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {

std::size_t gdn_projection_workspace_bytes(const GdnParameters& parameters, std::int32_t first,
                                           std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("GDN projection: invalid column interval");
    }
    if (const auto* single = std::get_if<LinearParameters>(&parameters.projection)) {
        const auto& w = single->weight;
        return ops::gdn_input_proj_workspace_capacity_bytes(w.qtype, w.n, w.k, single->policy,
                                                            first, last);
    }
    const auto* separate = std::get_if<ops::SeparateProjectionWeights>(&parameters.projection);
    if (separate != nullptr) {
        // One Linear per projection: the widest capacity covers all four.
        std::size_t bytes = 0;
        for (const auto& row : separate->rows) {
            const auto& w = row.weight;
            bytes = std::max(bytes, ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k,
                                                                         row.policy, first, last));
        }
        // The packed placement scratch: query/key/value project into it before their columns land
        // in the shared qkv parent at that parent's stride.
        std::size_t widest = 0;
        for (std::size_t row = 0; row < 3 && row < separate->rows.size(); ++row) {
            widest = std::max(widest, static_cast<std::size_t>(separate->rows[row].weight.n));
        }
        return std::max(bytes, widest * static_cast<std::size_t>(last) * sizeof(std::uint16_t));
    }
    const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection);
    return pair == nullptr ? 0
                           : ops::w4a8_gdn_input_workspace_capacity_bytes(
                                 pair->first.n, pair->second.n, pair->first.k, first, last);
}

std::size_t gdn_snapshot_workspace_bytes(const GdnParameters& parameters, const GdnConfig& config,
                                         std::int32_t batch, std::int32_t first_width,
                                         std::int32_t last_width) {
    const auto* single = std::get_if<LinearParameters>(&parameters.projection);
    std::size_t bytes;
    if (single && single->weight.qtype != QType::Q8_G32_FP16) {
        const auto& w = single->weight;
        bytes         = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            w.qtype, w.n, w.k, single->policy, batch, first_width, last_width);
    } else {
        bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.value_width()), batch, first_width, last_width);
    }
    // The native overlap contract takes a disjoint span even for a zero-scratch fused route.
    return std::max(std::size_t{1}, bytes);
}

std::size_t gdn_record_workspace_bytes(const GdnParameters& parameters, const GdnConfig& config,
                                       std::int32_t batch, std::int32_t first_width,
                                       std::int32_t last_width) {
    const auto* single = std::get_if<LinearParameters>(&parameters.projection);
    std::size_t bytes;
    if (single && single->weight.qtype != QType::Q8_G32_FP16) {
        const auto& w = single->weight;
        bytes         = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            w.qtype, w.n, w.k, single->policy, batch, first_width, last_width);
    } else {
        bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.value_width()), batch, first_width, last_width);
    }
    return std::max(std::size_t{1}, bytes);
}

void gdn_projection(const Tensor& hidden, const GdnParameters& parameters, Tensor& qkv, Tensor& z,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    const auto* separate = std::get_if<ops::SeparateProjectionWeights>(&parameters.projection);
    const auto* pair     = std::get_if<ops::PairedProjectionWeights>(&parameters.projection);
    if (separate != nullptr) {
        // rows holds query, key, value, z; the operator writes the first three into qkv in order.
        ops::gdn_input_proj(hidden, separate->rows[0].weight, separate->rows[1].weight,
                            separate->rows[2].weight, separate->rows[3].weight, qkv, z,
                            separate->rows[0].policy, workspace, stream);
    } else if (pair != nullptr) {
        ops::gdn_input_proj(hidden, pair->first, pair->second, qkv, z,
                            ops::LinearPolicy::AllowA8, workspace, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::gdn_input_proj(hidden, single.weight, qkv, z, single.policy, workspace, stream);
    }
}

void gdn_norm_control(const Tensor& residual, const Tensor& norm, float epsilon,
                      const GdnParameters& parameters, Tensor& hidden, Tensor& g, Tensor& beta,
                      WorkspaceArena& workspace, DeviceExecutionView execution) {
    ops::gdn_norm_control_proj(residual, norm, epsilon, parameters.control, parameters.a_log,
                               parameters.dt_bias, {hidden, {g, beta}}, workspace, execution);
}

void gdn_projection_snapshot(const Tensor& hidden, const GdnParameters& parameters,
                             const GdnConfig& config, Tensor& conv_states,
                             const Tensor& valid_columns, const Tensor& initial_slots,
                             const Tensor& destination_slots, Tensor& query, Tensor& key,
                             Tensor& value, Tensor& z, WorkspaceArena& workspace,
                             cudaStream_t stream) {
    auto scope = workspace.scope();
    WorkspaceArena scratch(workspace.alloc_bytes(gdn_snapshot_workspace_bytes(
        parameters, config, hidden.ne[2], hidden.ne[1], hidden.ne[1])));
    if (std::holds_alternative<ops::SeparateProjectionWeights>(parameters.projection)) {
        // The state-writing forms fuse the projection and the convolution step around one parent
        // row order; a per-projection artifact has no such order and would need its own composed
        // route. Refuse by name instead of reading the wrong rows.
        throw std::invalid_argument(
            "GDN projection snapshot: this artifact keeps q, k, v and z in their own parents; the "
            "state-writing snapshot form needs a fused parent");
    }
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj_conv_snapshot(hidden, pair->first, pair->second, parameters.convolution,
                                          conv_states, valid_columns, initial_slots,
                                          destination_slots, query, key, value, z, scratch, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::gdn_input_proj_conv_snapshot(
            hidden, single.weight, parameters.convolution, conv_states, valid_columns,
            initial_slots, destination_slots, query, key, value, z, single.policy, scratch, stream);
    }
}

void gdn_projection_record(const Tensor& hidden, const GdnParameters& parameters,
                           const GdnConfig& config, const Tensor& conv_states,
                           const Tensor& valid_columns, const Tensor& initial_slots,
                           Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                           Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    WorkspaceArena scratch(workspace.alloc_bytes(
        gdn_record_workspace_bytes(parameters, config, hidden.ne[2], hidden.ne[1], hidden.ne[1])));
    if (std::holds_alternative<ops::SeparateProjectionWeights>(parameters.projection)) {
        throw std::invalid_argument(
            "GDN projection record: this artifact keeps q, k, v and z in their own parents; the "
            "record-producing form needs a fused parent");
    }
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj_conv_record(hidden, pair->first, pair->second, parameters.convolution,
                                        conv_states, valid_columns, initial_slots, conv_record,
                                        query, key, value, z, scratch, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::gdn_input_proj_conv_record(hidden, single.weight, parameters.convolution, conv_states,
                                        valid_columns, initial_slots, conv_record, query, key,
                                        value, z, single.policy, scratch, stream);
    }
}

} // namespace ninfer::models::qwen3_5::execution
