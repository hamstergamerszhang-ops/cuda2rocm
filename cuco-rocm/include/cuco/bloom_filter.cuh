/*
 * Copyright 2026, rocm-cuda-compat contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! cuco::bloom_filter — GPU blocked Bloom filter for ROCm/HIP.
//!
//! Matches the cuco::bloom_filter API used by Sirius:
//!   bloom_filter<KeyT, Extent, Scope, Policy, Allocator>
//!
//! Implementation: blocked Bloom filter with k hash functions per key.
//! Each block is `words_per_block * sizeof(word_type)` bytes.
//! Insert: atomicOr on the bit array. Probe: read + AND.

#pragma once

#include "cuco/bloom_filter_policies.cuh"
#include "cuco/extent.cuh"
#include "cuco/hash_functions.cuh"

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace cuco {

// --- Device kernels (must be at namespace scope, not inside the class) ---

namespace detail {

template <typename WordT, typename Policy, typename KeyT, typename InputIt>
__global__ void bloom_add_kernel(WordT* bits, std::size_t num_blocks,
                                  std::size_t words_per_block,
                                  Policy policy, InputIt keys, std::size_t n) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  KeyT key = keys[idx];
  std::uint64_t hashes[16]; // max k=16
  std::size_t k = policy.num_hash_functions();
  assert(k <= 16 && "cuco::bloom_filter: num_hash_functions exceeds hardcoded max of 16");
  policy.hash(key, hashes, k);

  std::size_t block_idx = hashes[0] % num_blocks;
  WordT* block = bits + block_idx * words_per_block;

  std::size_t bits_per_word = sizeof(WordT) * 8;
  std::size_t bits_per_block = words_per_block * bits_per_word;

  #pragma unroll
  for (std::size_t i = 0; i < 16; ++i) {
    if (i >= k) break;
    std::size_t bit_idx = hashes[i] % bits_per_block;
    std::size_t word_idx = bit_idx / bits_per_word;
    std::size_t bit_offset = bit_idx % bits_per_word;
    atomicOr(&block[word_idx], WordT{1} << bit_offset);
  }
}

template <typename WordT, typename Policy, typename KeyT, typename InputIt, typename OutputIt>
__global__ void bloom_probe_kernel(WordT const* bits, std::size_t num_blocks,
                                    std::size_t words_per_block,
                                    Policy policy, InputIt keys, std::size_t n,
                                    OutputIt out) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  KeyT key = keys[idx];
  std::uint64_t hashes[16];
  std::size_t k = policy.num_hash_functions();
  assert(k <= 16 && "cuco::bloom_filter: num_hash_functions exceeds hardcoded max of 16");
  policy.hash(key, hashes, k);

  std::size_t block_idx = hashes[0] % num_blocks;
  WordT const* block = bits + block_idx * words_per_block;

  std::size_t bits_per_word = sizeof(WordT) * 8;
  std::size_t bits_per_block = words_per_block * bits_per_word;

  bool found = true;
  #pragma unroll
  for (std::size_t i = 0; i < 16; ++i) {
    if (i >= k) break;
    if (!found) break;
    std::size_t bit_idx = hashes[i] % bits_per_block;
    std::size_t word_idx = bit_idx / bits_per_word;
    std::size_t bit_offset = bit_idx % bits_per_word;
    if (!(block[word_idx] & (WordT{1} << bit_offset))) {
      found = false;
    }
  }
  out[idx] = found;
}

}  // namespace detail

template <typename KeyT,
          typename Extent = extent<std::size_t>,
          int Scope = 0,  // cuda::thread_scope_device (ignored on HIP)
          typename Policy = default_filter_policy<>,
          typename Allocator = dummy_allocator>
class bloom_filter {
 public:
  using key_type = KeyT;
  using word_type = typename Policy::word_type;
  using extent_type = Extent;
  using policy_type = Policy;
  using allocator_type = Allocator;

  static constexpr std::size_t words_per_block = Policy::words_per_block;
  static constexpr std::size_t bits_per_word = sizeof(word_type) * 8;
  static constexpr std::size_t bits_per_block = words_per_block * bits_per_word;

  /// Constructor: allocates `num_blocks * words_per_block * sizeof(word_type)` bytes.
  __host__ bloom_filter(Extent block_extent, Policy policy = {}, int = {},
                        Allocator alloc = {}, hipStream_t stream = 0)
    : block_extent_(static_cast<std::size_t>(block_extent)),
      policy_(policy),
      stream_(stream) {
    (void)alloc; // Allocator accepted for API compat, hipMalloc used
    std::size_t bytes = block_extent_ * words_per_block * sizeof(word_type);
    if (hipMalloc(&bits_, bytes) != hipSuccess) {
      throw std::runtime_error("cuco::bloom_filter: hipMalloc failed");
    }
    if (hipMemsetAsync(bits_, 0, bytes, stream) != hipSuccess) {
      hipFree(bits_);
      bits_ = nullptr;
      throw std::runtime_error("cuco::bloom_filter: hipMemsetAsync failed");
    }
  }

  __host__ ~bloom_filter() {
    if (bits_) {
      // Sync the construction/insert stream before freeing — hipFree does
      // NOT sync non-default streams, so async ops on stream_ could still
      // be reading/writing bits_ when we free it.
      hipStreamSynchronize(stream_);
      hipFree(bits_);
    }
  }

  // Non-copyable, non-movable (owns device memory)
  bloom_filter(bloom_filter const&) = delete;
  bloom_filter& operator=(bloom_filter const&) = delete;

  /// Number of blocks in the filter.
  __host__ __device__ std::size_t block_extent() const { return block_extent_; }

  /// Pointer to the device bit array.
  __host__ __device__ word_type* data() { return bits_; }
  __host__ __device__ word_type const* data() const { return bits_; }

  /// Async insert a range of keys.
  template <typename InputIt>
  __host__ void add_async(InputIt first, InputIt last, hipStream_t stream) {
    auto n = static_cast<std::size_t>(last - first);
    if (n == 0) return;
    std::size_t block = 256;
    std::size_t grid = (n + block - 1) / block;
    detail::bloom_add_kernel<word_type, Policy, KeyT, InputIt>
      <<<grid, block, 0, stream>>>(bits_, block_extent_, words_per_block,
                                    policy_, first, n);
  }

  /// Async probe: write `true`/`false` for each key.
  template <typename InputIt, typename OutputIt>
  __host__ void contains_async(InputIt first, InputIt last, OutputIt out, hipStream_t stream) {
    auto n = static_cast<std::size_t>(last - first);
    if (n == 0) return;
    std::size_t block = 256;
    std::size_t grid = (n + block - 1) / block;
    detail::bloom_probe_kernel<word_type, Policy, KeyT, InputIt, OutputIt>
      <<<grid, block, 0, stream>>>(bits_, block_extent_, words_per_block,
                                    policy_, first, n, out);
  }

 private:
  word_type* bits_{nullptr};
  std::size_t block_extent_{};
  Policy policy_{};
  hipStream_t stream_{0};
};

}  // namespace cuco
