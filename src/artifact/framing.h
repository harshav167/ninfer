#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace ninfer::artifact {

inline constexpr std::array<std::byte, 8> kEntryMagic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{3},
};
inline constexpr std::array<std::byte, 8> kPartMagic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'P'},
    std::byte{'R'}, std::byte{'T'}, std::byte{0},   std::byte{3},
};
inline constexpr std::size_t kHeaderBytes        = 32;
inline constexpr std::uint64_t kPayloadAlignment = 4096;
using ArtifactId                                 = std::array<std::byte, 16>;

// Canonical text form of the container identity: the 16 header bytes as 32 lowercase hex digits.
[[nodiscard]] inline std::string artifact_id_string(const ArtifactId& id) {
    constexpr char digits[] = "0123456789abcdef";
    std::string out(id.size() * 2, '\0');
    for (std::size_t i = 0; i < id.size(); ++i) {
        const auto byte = std::to_integer<unsigned>(id[i]);
        out[2 * i]      = digits[byte >> 4];
        out[2 * i + 1]  = digits[byte & 15];
    }
    return out;
}

inline std::uint64_t read_u64_le(const std::byte* bytes) noexcept {
    std::uint64_t out = 0;
    for (unsigned i = 0; i < 8; ++i) {
        out |= std::uint64_t(std::to_integer<unsigned>(bytes[i])) << (8 * i);
    }
    return out;
}

inline std::uint32_t read_u32_le(const std::byte* bytes) noexcept {
    std::uint32_t out = 0;
    for (unsigned i = 0; i < 4; ++i) {
        out |= std::uint32_t(std::to_integer<unsigned>(bytes[i])) << (8 * i);
    }
    return out;
}

} // namespace ninfer::artifact
