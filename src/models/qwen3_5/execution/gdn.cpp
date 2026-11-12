#include "models/qwen3_5/execution/gdn.h"

#include "ninfer/ops/gdn_control.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/w4a8.h"
// The separate-parent record route finishes with the projected-conv kernel the composed record
// routes inside gdn_input_proj already use (`compose_record` in src/ops/wrapper/gdn_input_proj.cpp
// calls exactly this launch). It stays private to ninfer_ops because the fused parents reach it
// through that wrapper; the model layer needs it because a separate-parent artifact has no fused
// parent to route through. Same kernel, same publish-less record form, so the record and snapshot
// outputs stay bit-identical by construction rather than by a second implementation agreeing.
#include "ops/gdn_input_proj/gdn_projected_conv.h"

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
        const std::int32_t widest = std::max(
            std::max<std::int32_t>(separate->rows[0].weight.n, separate->rows[1].weight.n),
            separate->rows[2].weight.n);
        return std::max(bytes, ops::gdn_input_proj_placement_workspace_bytes(widest, last));
    }
    const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection);
    return pair == nullptr ? 0
                           : ops::w4a8_gdn_input_workspace_capacity_bytes(
                                 pair->first.n, pair->second.n, pair->first.k, first, last);
}

std::size_t gdn_snapshot_workspace_bytes(const GdnParameters& parameters, const GdnConfig& config,
                                         std::int32_t batch, std::int32_t first_width,
                                         std::int32_t last_width) {
    if (const auto* separate = std::get_if<ops::SeparateProjectionWeights>(&parameters.projection)) {
        // Composed route: packed projection rectangle plus the projection's placement scratch, both
        // live at once; the four Linear calls may add scratch of their own on top.
        std::size_t linear_bytes = 0;
        for (const auto& row : separate->rows) {
            linear_bytes = std::max(linear_bytes,
                                    ops::linear_workspace_capacity_bytes(
                                        row.weight.qtype, row.weight.n, row.weight.k, row.policy,
                                        first_width, last_width));
        }
        return ops::gdn_input_proj_conv_snapshot_separate_workspace_capacity_bytes(
                   separate->rows[0].weight.n, separate->rows[1].weight.n, separate->rows[2].weight.n,
                   batch, first_width, last_width) +
               linear_bytes;
    }
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
    if (const auto* separate = std::get_if<ops::SeparateProjectionWeights>(&parameters.projection)) {
        // Composition route: the projection writes conv_record at that record's own channel stride
        // through the placement scratch of gdn_input_proj's four-parent form, and each of those
        // four Linear calls may add its own transient bytes (zero for the groupwise formats,
        // nonzero for the block-scaled ones). Both are live at the same time. The convolution that
        // follows needs no arena at all - it reads conv_record and the state pool in place.
        const std::int32_t columns = batch * last_width;
        std::size_t linear_bytes   = 0;
        for (const auto& row : separate->rows) {
            linear_bytes =
                std::max(linear_bytes, ops::linear_workspace_capacity_bytes(
                                           row.weight.qtype, row.weight.n, row.weight.k, row.policy,
                                           first_width, last_width));
        }
        const std::int32_t widest = std::max(
            std::max<std::int32_t>(separate->rows[0].weight.n, separate->rows[1].weight.n),
            separate->rows[2].weight.n);
        return std::max(std::size_t{1},
                        ops::gdn_input_proj_placement_workspace_bytes(widest, columns) +
                            linear_bytes);
    }
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
    if (const auto* separate = std::get_if<ops::SeparateProjectionWeights>(&parameters.projection)) {
        // No fused parent row order exists here, so the state-writing form cannot fuse the
        // projection into the convolution. It runs as the composition instead: project the
        // rectangle, then convolve it and publish the destination history. The row windows below
        // (query/output_gate and friends) are already packed per projection, which is exactly what
        // that composition writes.
        ops::gdn_input_proj_conv_snapshot(hidden, separate->rows[0].weight, separate->rows[1].weight,
                                          separate->rows[2].weight, separate->rows[3].weight,
                                          parameters.convolution, conv_states, valid_columns,
                                          initial_slots, destination_slots, query, key, value, z,
                                          separate->rows[0].policy, scratch, stream);
        return;
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
    if (const auto* separate = std::get_if<ops::SeparateProjectionWeights>(&parameters.projection)) {
        // Separate parents carry no fused row order, so the record-producing form runs as the
        // composition the fused routes fall back to on their non-fused widths: project the whole
        // [C, T*B] rectangle into conv_record itself - which is exactly the packed record plane the
        // contract defines, oldest tap first, one column per physical token - then run the same
        // width-four convolution that the state-writing form runs, with the publish arm that leaves
        // the history alone. Z is projected for every physical column, like the fused forms.
        //
        // Numerically this is not a second implementation: the projection is the four-parent
        // gdn_input_proj the prefill and snapshot routes already call, and the convolution is
        // gdn_projected_conv_kernel with NoHistoryPublish - the template instance
        // gdn_projection_snapshot reaches through the snapshot form. Record and snapshot therefore
        // agree bit-for-bit from the same initial history, which is the record Op's contract.
        const std::int32_t width   = hidden.ne[1];
        const std::int32_t batch   = hidden.ne[2];
        const std::int32_t columns = width * batch;
        const std::int32_t channels = separate->rows[0].weight.n + separate->rows[1].weight.n +
                                      separate->rows[2].weight.n;
        if (conv_record.ne[0] != channels || conv_record.ne[1] != width ||
            conv_record.ne[2] != batch) {
            throw std::invalid_argument(
                "GDN projection record: the record plane must be [query+key+value, T, B]");
        }
        Tensor hidden_flat(hidden.data, hidden.dtype, {hidden.ne[0], columns});
        Tensor record_flat(conv_record.data, conv_record.dtype, {channels, columns});
        Tensor gate_flat(z.data, z.dtype, {separate->rows[3].weight.n, columns});
        ops::gdn_input_proj(hidden_flat, separate->rows[0].weight, separate->rows[1].weight,
                            separate->rows[2].weight, separate->rows[3].weight, record_flat,
                            gate_flat, separate->rows[0].policy, scratch, stream);
        ops::detail::gdn_projected_conv_record_launch(conv_record, parameters.convolution,
                                                      conv_states, valid_columns, initial_slots,
                                                      query, key, value, stream);
        return;
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
