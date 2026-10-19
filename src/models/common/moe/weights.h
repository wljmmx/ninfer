#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/expert_cache.h"
#include "core/weight_bank.h"

namespace ninfer::models::moe {

class ExpertHostTier;

struct ExpertBinding {
    const artifact::ParameterReference& gate;
    const artifact::ParameterReference& up;
    const artifact::ParameterReference& down;
};

struct MatrixSource {
    artifact::ObjectHandle object;
    WeightBankLayout layout;
};

struct BankDescription {
    MatrixSource gate_up, down;
};

// Logical expert order must form affine gate/up and down banks in the stored parents.
[[nodiscard]] BankDescription describe_bank(const artifact::Reader&,
                                            std::span<const ExpertBinding>);
// Three rank-3 logical [E,I,H], [E,I,H], [E,H,I] bindings over affine stored banks.
[[nodiscard]] BankDescription describe_bank(const artifact::Reader&, const ExpertBinding&);
[[nodiscard]] std::vector<ExpertBankSource> bind_banks(std::span<const BankDescription>,
                                                       const artifact::MaterializedArtifact&,
                                                       std::span<const Device>,
                                                       const ExpertHostTier* host = nullptr);
// port (sm_89): one upload image per expert, addressed directly in the artifact payload.
// Images follow the cache's own entry layout, so a fetched expert uploads with one copy.
[[nodiscard]] std::vector<ExpertFileBank> describe_expert_files(
    const artifact::Reader&, std::span<const BankDescription>);

// Bridges artifact payload reads to the cache's file tier. Opens its own read-only handles, so
// it outlives the load-time Reader. Synchronous, host side, never inside capture.
class ExpertFileReader final : public ExpertBlobReader {
public:
    explicit ExpertFileReader(const artifact::Reader& reader);
    void read_entry(const ExpertFileEntry&, std::span<std::byte>) const override;

private:
    std::vector<std::unique_ptr<io::ReadOnlyFile>> files_;
};

} // namespace ninfer::models::moe
