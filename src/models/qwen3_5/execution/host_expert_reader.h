#pragma once

#include "core/expert_file.h"
#include "models/qwen3_5/model.h"

#include <span>

namespace ninfer::models::qwen3_5::execution {

// port (sm_89): expert cache source backed by the retained Host replica. Chunks produced by
// describe_fused_expert_host carry the object index and the payload offset, so a hit copies from
// RAM instead of reading the artifact again.
class HostExpertReader final : public ExpertBlobReader {
public:
    explicit HostExpertReader(const Model& model) : model_(model) {}

    void read_entry(const ExpertFileEntry& entry, std::span<std::byte> destination) const override {
        for (const auto& chunk : entry.chunks) {
            const auto bytes = model_.host_bytes(artifact::ObjectHandle{chunk.file_index});
            if (std::getenv("NINFER_EXPERT_TRACE")) {
                std::fprintf(stderr, "host chunk obj=%u off=%llu bytes=%llu image=%llu host=%zu\\n",
                             chunk.file_index, static_cast<unsigned long long>(chunk.file_offset),
                             static_cast<unsigned long long>(chunk.bytes),
                             static_cast<unsigned long long>(chunk.image_offset), bytes.size());
            }
            if (chunk.file_offset > bytes.size() ||
                chunk.bytes > bytes.size() - chunk.file_offset ||
                chunk.image_offset > entry.image_bytes ||
                chunk.bytes > entry.image_bytes - chunk.image_offset) {
                throw std::runtime_error("host expert chunk lies outside its object");
            }
            std::memcpy(destination.data() + chunk.image_offset,
                        bytes.data() + chunk.file_offset, static_cast<std::size_t>(chunk.bytes));
        }
    }

private:
    const Model& model_;
};

} // namespace ninfer::models::qwen3_5::execution