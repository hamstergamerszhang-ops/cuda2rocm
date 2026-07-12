/*
 * Copyright 2026, rocm-cuda-compat contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! cuco hash functions: xxhash_64, default_hash_function.
//! xxHash implementation ported from the reference C code (BSD-2-Clause).

#pragma once

#include <cstdint>
#include <cstddef>

namespace cuco {

namespace detail {

// xxHash 64-bit — device-compatible implementation.
// Reference: https://github.com/Cyan4973/xxHash (BSD-2-Clause)
__host__ __device__ inline std::uint64_t xxh64(void const* input, std::size_t len,
                                                std::uint64_t seed = 0) {
  constexpr std::uint64_t PRIME1 = 0x9E3779B185EBCA87ULL;
  constexpr std::uint64_t PRIME2 = 0xC2B2AE3D27D4EB4FULL;
  constexpr std::uint64_t PRIME3 = 0x165667B19E3779F9ULL;
  constexpr std::uint64_t PRIME4 = 0x85EBCA77C2B2AE63ULL;
  constexpr std::uint64_t PRIME5 = 0x27D4EB2F165667C5ULL;

  auto const* p = static_cast<std::uint8_t const*>(input);
  auto const* end = p + len;
  std::uint64_t h;

  auto rotl = [](std::uint64_t x, int r) -> std::uint64_t {
    return (x << r) | (x >> (64 - r));
  };

  if (len >= 32) {
    auto const* limit = end - 32;
    std::uint64_t v1 = seed + PRIME1 + PRIME2;
    std::uint64_t v2 = seed + PRIME2;
    std::uint64_t v3 = seed;
    std::uint64_t v4 = seed - PRIME1;
    do {
      auto read64 = [](std::uint8_t const* d) -> std::uint64_t {
        return static_cast<std::uint64_t>(d[0]) |
               (static_cast<std::uint64_t>(d[1]) << 8) |
               (static_cast<std::uint64_t>(d[2]) << 16) |
               (static_cast<std::uint64_t>(d[3]) << 24) |
               (static_cast<std::uint64_t>(d[4]) << 32) |
               (static_cast<std::uint64_t>(d[5]) << 40) |
               (static_cast<std::uint64_t>(d[6]) << 48) |
               (static_cast<std::uint64_t>(d[7]) << 56);
      };
      v1 = rotl(v1 + read64(p) * PRIME2, 31) * PRIME1; p += 8;
      v2 = rotl(v2 + read64(p) * PRIME2, 31) * PRIME1; p += 8;
      v3 = rotl(v3 + read64(p) * PRIME2, 31) * PRIME1; p += 8;
      v4 = rotl(v4 + read64(p) * PRIME2, 31) * PRIME1; p += 8;
    } while (p <= limit);
    h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
    h = (h ^ (rotl(v1 * PRIME2, 31) * PRIME1)) * PRIME1 + PRIME4;
    h = (h ^ (rotl(v2 * PRIME2, 31) * PRIME1)) * PRIME1 + PRIME4;
    h = (h ^ (rotl(v3 * PRIME2, 31) * PRIME1)) * PRIME1 + PRIME4;
    h = (h ^ (rotl(v4 * PRIME2, 31) * PRIME1)) * PRIME1 + PRIME4;
  } else {
    h = seed + PRIME5;
  }
  h += static_cast<std::uint64_t>(len);

  while (p + 8 <= end) {
    auto read64 = [](std::uint8_t const* d) -> std::uint64_t {
      return static_cast<std::uint64_t>(d[0]) |
             (static_cast<std::uint64_t>(d[1]) << 8) |
             (static_cast<std::uint64_t>(d[2]) << 16) |
             (static_cast<std::uint64_t>(d[3]) << 24) |
             (static_cast<std::uint64_t>(d[4]) << 32) |
             (static_cast<std::uint64_t>(d[5]) << 40) |
             (static_cast<std::uint64_t>(d[6]) << 48) |
             (static_cast<std::uint64_t>(d[7]) << 56);
    };
    std::uint64_t k1 = rotl(read64(p) * PRIME2, 31) * PRIME1;
    h = rotl(h ^ k1, 27) * PRIME1 + PRIME4;
    p += 8;
  }
  if (p + 4 <= end) {
    auto read32 = [](std::uint8_t const* d) -> std::uint32_t {
      return static_cast<std::uint32_t>(d[0]) |
             (static_cast<std::uint32_t>(d[1]) << 8) |
             (static_cast<std::uint32_t>(d[2]) << 16) |
             (static_cast<std::uint32_t>(d[3]) << 24);
    };
    h ^= static_cast<std::uint64_t>(read32(p)) * PRIME1;
    h = rotl(h, 23) * PRIME2 + PRIME3;
    p += 4;
  }
  while (p < end) {
    h ^= static_cast<std::uint64_t>(*p) * PRIME5;
    h = rotl(h, 11) * PRIME1;
    ++p;
  }
  h ^= h >> 33; h *= PRIME2;
  h ^= h >> 29; h *= PRIME3;
  h ^= h >> 32;
  return h;
}

}  // namespace detail

/// xxHash 64-bit functor. Matches cuco::xxhash_64<KeyT>.
template <typename KeyT>
struct xxhash_64 {
  __host__ __device__ std::uint64_t operator()(KeyT const& key) const {
    return detail::xxh64(&key, sizeof(KeyT));
  }
};

/// Default hash function — uses xxhash_64.
template <typename KeyT>
struct default_hash_function {
  __host__ __device__ std::uint64_t operator()(KeyT const& key) const {
    return xxhash_64<KeyT>{}(key);
  }
};

}  // namespace cuco
