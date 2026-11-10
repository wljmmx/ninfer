#pragma once

#include "cache_reference.h"
#include "e8_reference.h"
#include "ops/op_tester.h"

namespace ninfer::test::qsa::sparse {

using prepare::is_e8;

// The sparse-Op suite keeps filling the E8 planes from a deterministic code pattern rather than
// through the encoders: its oracle then compares the reader against exactly the bytes on the card,
// independent of whether prepare and append agree about how to produce them.

struct EncodedPlane {
    int leading, element;
    std::int64_t row, head, page;
    std::vector<std::uint8_t> host;
    GuardedDeviceBuffer device;
    Tensor view;

    EncodedPlane(int features, DType dtype, int pages, bool padded, int heads = 2)
        : leading(padded ? dtype_size(dtype) : 0), element(dtype_size(dtype)),
          row(features * element + (padded ? element : 0)), head(64 * row + leading),
          page(heads * head + leading), host(leading + page * pages, 0xcd), device(host.size()),
          view(static_cast<std::uint8_t*>(device.data()) + leading, dtype,
               {features, 64, heads, pages}) {
        view.nb[1] = row;
        view.nb[2] = head;
        view.nb[3] = page;
    }

    std::size_t offset(int r, int h, int p) const {
        return leading + r * row + h * head + p * page;
    }

    void upload() { device.copy_from_host(host.data(), host.size()); }

    int preserved(const std::string& label) const {
        return verify_exact(label.c_str(), from_device<std::uint8_t>(device.data(), host.size()),
                            host) +
               device.verify_guards(label);
    }
};

struct EncodedVector {
    prepare::VectorEncoding format;
    EncodedPlane data, scales;

    EncodedVector(prepare::VectorEncoding fmt, int pages, bool padded, bool padded_scales = false,
                  int heads = 2)
        : format(fmt), data(fmt.data_extent, fmt.data_dtype, pages, padded, heads),
          scales(fmt.scale_extent, fmt.scale_dtype, pages, padded || padded_scales, heads) {}

    template <class Source>
    void encode(const Source& source) {
        std::fill(data.host.begin(), data.host.end(), 0xcd);
        std::fill(scales.host.begin(), scales.host.end(), 0xcd);
        for (int p = 0; p < data.view.ne[3]; ++p)
            for (int h = 0; h < data.view.ne[2]; ++h)
                for (int r = 0; r < 64; ++r) {
                    const auto db = data.offset(r, h, p), sb = scales.offset(r, h, p);
                    if (is_e8(format.encoding)) {
                        pattern(db, sb, r, h, p);
                        continue;
                    }
                    const auto first   = source.host[source.offset(0, h, r, p)];
                    const auto initial = source.view.dtype == DType::BF16
                                             ? bf16_to_f32(first)
                                             : prepare::fp16_decode(first);
                    if (!std::isfinite(initial)) continue;
                    std::array<float, 256> values;
                    for (int d = 0; d < 256; ++d) {
                        const auto word = source.host[source.offset(d, h, r, p)];
                        values[d]       = source.view.dtype == DType::BF16 ? bf16_to_f32(word)
                                                                           : prepare::fp16_decode(word);
                    }
                    const auto encoded = prepare::encode_row(values, format);
                    std::copy(encoded.data.begin(), encoded.data.end(),
                              data.host.begin() + data.offset(r, h, p));
                    std::copy(encoded.scales.begin(), encoded.scales.end(),
                              scales.host.begin() + scales.offset(r, h, p));
                }
        data.upload();
        scales.upload();
    }

    // A deterministic E8 row: eight real root codes per group with live radii, plus residual bits
    // on the 96-byte carrier.
    void pattern(std::size_t db, std::size_t sb, int r, int h, int p) {
        static constexpr std::uint8_t kRoots[8]{0, 3, 17, 64, 112, 141, 200, 240};
        const bool residual = format.encoding == prepare::Encoding::E8Cylinder;
        for (int group = 0; group < 4; ++group) {
            for (int s = 0; s < 8; ++s) {
                const int seed = (r + h + p + group * 8 + s) % 8;
                data.host[db + (group * 8 + s) * 2]     = kRoots[seed];
                const int radius_index                  = 1 + (s % 3);
                const int axis_index                    = (group * 5 + s * 3) & 0x0F;
                data.host[db + (group * 8 + s) * 2 + 1] =
                    std::uint8_t((radius_index << 4) | axis_index);
                if (residual) {
                    data.host[db + 64 + group * 8 + s] =
                        std::uint8_t(0x5A ^ (seed * 37 + group * 11 + r));
                }
            }
            prepare::write_u16(scales.host, sb + 2 * group,
                               prepare::fp16_encode(0.02F + 0.01F * float(group)));
        }
    }

    double decode(int d, int h, int r, int p) const {
        const auto db = data.offset(r, h, p), sb = scales.offset(r, h, p);
        const int group = d / format.group;
        const double scale = prepare::fp16_decode(prepare::read_u16(scales.host, sb + 2 * group));
        if (is_e8(format.encoding)) {
            const int subspace = (d / 64) * 8 + ((d % 64) / 8);
            std::array<int, 8> decoded{};
            e8::decode_8d(data.host[db + subspace * 2], data.host[db + subspace * 2 + 1],
                          !format.rotate, decoded);
            int code = decoded[d % 8];
            if (format.encoding == prepare::Encoding::E8Cylinder) {
                const int sign = (data.host[db + 64 + subspace] >> (d % 8)) & 1;
                code           = 4 * (code + (sign ? 1 : -1));
            }
            return double(code) * scale;
        }
        const bool fp4    = format.encoding == prepare::Encoding::Nvfp4;
        const bool nibble = fp4 || format.encoding == prepare::Encoding::Int4;
        const auto code =
            nibble ? (data.host[db + d / 2] >> (4 * (d % 2))) & 15 : data.host[db + d];
        const double row_scale =
            fp4 ? prepare::positive_fp8(scales.host[sb + group]) : scale;
        return prepare::decode_code(code, format.encoding) * row_scale;
    }
};

} // namespace ninfer::test::qsa::sparse
