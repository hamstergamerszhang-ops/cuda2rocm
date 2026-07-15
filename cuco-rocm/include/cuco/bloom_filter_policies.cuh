/*
 * Copyright 2026, rocm-cuda-compat contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! cuco Bloom filter policies: arrow_filter_policy, default_filter_policy.

#pragma once

#include "cuco/hash_functions.cuh"
#include <cstdint>

namespace cuco {

/// Default Bloom filter policy: k hash functions derived from double hashing.
template <typename Hash = xxhash_64<std::uint64_t>,
          typename WordT = std::uint32_t,
          std::size_t WordsPerBlock = 8>
struct default_filter_policy {
  using hash_type = Hash;
  using word_type = WordT;
  static constexpr std::size_t words_per_block = WordsPerBlock;

  Hash hash_;

  __host__ __device__ default_filter_policy() = default;

  /// Number of hash functions (optimal ~7 for 16 bits/key).
  __host__ __device__ std::size_t num_hash_functions() const { return 7; }

  /// Returns k hash values for the key, writing into the output range.
  template <typename KeyT, typename OutputIt>
  __host__ __device__ void hash(KeyT const& key, OutputIt out, std::size_t k) const {
    std::uint64_t h1 = hash_(key);
    std::uint64_t h2 = (h1 >> 32) | 1;  // ensure non-zero stride (guard against h2=0 degeneration)
    for (std::size_t i = 0; i < k; ++i) {
      out[i] = h1 + i * h2;
    }
  }
};

/// Arrow filter policy: optimized for block-aligned Bloom filters.
/// Matches cuco::arrow_filter_policy<KeyT> used by Sirius.
template <typename KeyT>
struct arrow_filter_policy {
  using word_type = std::uint32_t;
  static constexpr std::size_t words_per_block = 8;
  static constexpr std::size_t max_filter_blocks = (1u << 31) / (words_per_block * sizeof(word_type));

  __host__ __device__ arrow_filter_policy() = default;

  /// Number of hash functions.
  __host__ __device__ std::size_t num_hash_functions() const { return 7; }

  template <typename OutputIt>
  __host__ __device__ void hash(KeyT const& key, OutputIt out, std::size_t k) const {
    xxhash_64<KeyT> h;
    std::uint64_t h1 = h(key);
    std::uint64_t h2 = h1 * 0x9E3779B97F4A7C15ULL;
    // Block-aligned: words_per_block=8, word_type=uint32_t → 32-byte blocks.
    // Shift by log2(32) = 5 to get a block index. (>> 6 would assume 64-byte
    // blocks and leave half the filter's blocks unreachable, raising the
    // false-positive rate.)
    constexpr std::size_t block_shift = 5;
    for (std::size_t i = 0; i < k; ++i) {
      out[i] = (h1 + i * h2) >> block_shift;
    }
  }
};

}  // namespace cuco
