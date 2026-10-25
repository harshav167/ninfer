#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] std::string format_pretty_bytes(std::uint64_t bytes);
[[nodiscard]] std::string format_pretty_count(std::uint64_t count);
[[nodiscard]] std::string format_pretty_duration(double seconds);
[[nodiscard]] std::string format_pretty_percent(double ratio);
[[nodiscard]] std::string format_pretty_rate(double per_second, std::string_view unit);
[[nodiscard]] std::string format_pretty_text(std::string_view value);
// Compact index list: strictly consecutive runs become "a..b", runs are joined with ','.
[[nodiscard]] std::string format_pretty_index_runs(std::span<const std::uint32_t> indices);

} // namespace ninfer::product
