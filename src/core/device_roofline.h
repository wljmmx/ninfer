#pragma once
//
// device_roofline.h - the device's nominal DRAM bandwidth, queried instead of assumed.
//
// CUDA 13 removed cudaDeviceProp::memoryClockRate (the field is gone, not merely deprecated), so
// the bandwidth that used to fall out of the struct has to come from the attributes that remain:
// cudaDevAttrMemoryClockRate and cudaDevAttrGlobalMemoryBusWidth. A hard-coded roofline is only
// correct for the one board it was written for; every other device then divides by the wrong
// denominator (on an RTX 4090 against 1792 GB/s every reported fraction is ~44% too low).
//
// When the attributes are unavailable this reports 0 and callers must print "n/a" rather than
// invent a number: an unknown roofline is information, a wrong one is not.

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::core {

struct DeviceRoofline {
    double       dram_gbs             = 0.0; // 0 = unknown; never guess
    int          multiprocessor_count = 0;
    int          compute_major        = 0;
    int          compute_minor        = 0;
    std::int32_t memory_clock_khz     = 0;
    std::int32_t memory_bus_bits      = 0;
};

// GDDR/HBM move two transfers per memory clock (DDR), so
//   bytes/s = 2 * clock_hz * bus_bits / 8   and with clock in kHz this is
//   GB/s    = clock_khz * bus_bits / 4e6.
// Checked against published peaks: 4090 10501 MHz x 384 bit = 1008 GB/s,
// 5090 14001 MHz x 512 bit = 1792 GB/s, A100 1215 MHz x 5120 bit = 1555 GB/s.
[[nodiscard]] inline double dram_gbs_from_memory_clock(std::int32_t clock_khz,
                                                       std::int32_t bus_bits) noexcept {
    if (clock_khz <= 0 || bus_bits <= 0) { return 0.0; }
    return static_cast<double>(clock_khz) * static_cast<double>(bus_bits) / 4.0e6;
}

// Naming only; nothing in the engine branches on it. Kept here so a bench report can say which
// architecture its numbers belong to instead of implying "5090" everywhere.
[[nodiscard]] inline const char* architecture_name(int major, int minor) noexcept {
    switch (major) {
    case 7: return "Volta/Turing";
    case 8: return minor == 9 ? "Ada (sm_89)" : "Ampere";
    case 9: return "Hopper";
    case 10: return "Blackwell";
    case 12: return "Blackwell GeForce";
    default: return "unknown";
    }
}

[[nodiscard]] inline DeviceRoofline query_device_roofline(int device) noexcept {
    DeviceRoofline out;
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, device) != cudaSuccess) { return out; }
    out.multiprocessor_count = props.multiProcessorCount;
    out.compute_major        = props.major;
    out.compute_minor        = props.minor;

    std::int32_t clock_khz = 0;
    std::int32_t bus_bits  = 0;
    if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, device) != cudaSuccess) {
        clock_khz = 0;
    }
    if (cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, device) != cudaSuccess) {
        bus_bits = 0;
    }
    out.memory_clock_khz = clock_khz;
    out.memory_bus_bits  = bus_bits;
    out.dram_gbs         = dram_gbs_from_memory_clock(clock_khz, bus_bits);
    return out;
}

} // namespace ninfer::core
