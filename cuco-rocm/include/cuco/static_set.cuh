/*
 * Copyright 2026, rocm-cuda-compat contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! cuco::static_set — GPU open-addressing hash set for ROCm/HIP.
//!
//! Matches the cuco::static_set API used by Sirius.

#pragma once

#include "cuco/extent.cuh"
#include "cuco/hash_functions.cuh"
#include "cuco/operator.hpp"
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace cuco {

/// Sentinel wrapper for empty keys.
template <typename KeyT>
struct empty_key {
  KeyT value;
  __host__ __device__ empty_key(KeyT v) : value(v) {}
  __host__ __device__ operator KeyT() const { return value; }
};

/// Double hashing probe strategy.
template <int CgSize, typename Hash>
struct double_hashing {
  Hash hash_;
  __host__ __device__ double_hashing() = default;

  template <typename KeyT>
  __host__ __device__ std::size_t operator()(KeyT const& key, std::size_t probe) const {
    std::uint64_t h = hash_(key);
    std::uint64_t h1 = h;
    std::uint64_t h2 = (h >> 32) | 1;  // ensure non-zero stride
    return static_cast<std::size_t>((h1 + probe * h2));
  }
};

// contains_t and contains are defined in operator.hpp

namespace detail {

template <typename KeyT, typename Hash, typename InputIt>
__global__ void set_insert_kernel(KeyT* slots, std::size_t capacity,
                                   KeyT empty_key, Hash hash,
                                   InputIt keys, std::size_t n) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  KeyT key = keys[idx];
  for (std::size_t probe = 0; probe < capacity; ++probe) {
    std::size_t slot_idx = hash(key, probe) % capacity;
    KeyT old = atomicCAS(&slots[slot_idx], empty_key, key);
    if (old == empty_key || old == key) return;
  }
}

}  // namespace detail

/// Lightweight view for probing (device-constructible).
template <typename KeyT, typename Hash>
struct set_ref {
  KeyT const* slots;
  std::size_t capacity;
  KeyT empty_key;
  Hash hash;

  __host__ __device__ bool operator()(KeyT const& key) const {
    for (std::size_t probe = 0; probe < capacity; ++probe) {
      std::size_t idx = hash(key, probe) % capacity;
      KeyT slot = slots[idx];
      if (slot == key) return true;
      if (slot == empty_key) return false;
    }
    return false;
  }
};

/// GPU open-addressing hash set.
template <typename KeyT,
          typename Extent = extent<std::size_t>,
          int Scope = 0,
          typename Pred = cuco::dummy_allocator,
          typename Hash = double_hashing<1, default_hash_function<KeyT>>,
          typename Allocator = cuco::dummy_allocator>
class static_set {
 public:
  using key_type = KeyT;
  using extent_type = Extent;
  using hasher_type = Hash;
  using allocator_type = Allocator;

  __host__ static_set(Extent capacity, empty_key<KeyT> empty, Pred = {}, int = {},
                      int = {}, Allocator alloc = {}, hipStream_t stream = 0)
    : capacity_(static_cast<std::size_t>(capacity)),
      empty_key_(empty.value),
      hash_{} {
    (void)alloc; // Allocator accepted for API compat, hipMalloc used
    std::size_t bytes = capacity_ * sizeof(KeyT);
    hipMalloc(&slots_, bytes);
    // Fill with empty key: use a kernel or hipMemset for byte-sized keys,
    // otherwise copy from a host buffer.
    fill_empty_key<<<1, 1>>>(slots_, empty_key_, capacity_);
  }

  __host__ ~static_set() {
    if (slots_) hipFree(slots_);
  }

  static_set(static_set const&) = delete;
  static_set& operator=(static_set const&) = delete;

  __host__ __device__ std::size_t capacity() const { return capacity_; }
  __host__ __device__ KeyT* data() { return slots_; }
  __host__ __device__ KeyT const* data() const { return slots_; }

  /// Async insert a range of keys.
  template <typename InputIt>
  __host__ void insert_async(InputIt first, InputIt last, hipStream_t stream) {
    auto n = static_cast<std::size_t>(last - first);
    if (n == 0) return;
    std::size_t block = 256;
    std::size_t grid = (n + block - 1) / block;
    detail::set_insert_kernel<KeyT, Hash, InputIt>
      <<<grid, block, 0, stream>>>(slots_, capacity_, empty_key_, hash_, first, n);
  }

  /// Returns a reference object for contains queries.
  __host__ __device__ set_ref<KeyT, Hash> ref(contains_t) const {
    return set_ref<KeyT, Hash>{slots_, capacity_, empty_key_, hash_};
  }

 private:
  KeyT* slots_{nullptr};
  std::size_t capacity_{};
  KeyT empty_key_{};
  Hash hash_{};

  /// Fill slots with empty key using a simple kernel.
  __global__ static void fill_empty_key(KeyT* slots, KeyT empty, std::size_t n) {
    for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) {
      slots[i] = empty;
    }
  }
};

}  // namespace cuco
