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
#include <stdexcept>

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

template <typename KeyT>
__global__ void fill_empty_key_kernel(KeyT* slots, KeyT empty, std::size_t n) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    slots[i] = empty;
  }
}

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

template <typename KeyT, typename Hash, typename InputIt, typename OutputIt>
__global__ void set_probe_kernel(KeyT const* slots, std::size_t capacity,
                                   KeyT empty_key, Hash hash,
                                   InputIt keys, std::size_t n, OutputIt out) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  KeyT key = keys[idx];
  bool found = false;
  for (std::size_t probe = 0; probe < capacity; ++probe) {
    std::size_t slot_idx = hash(key, probe) % capacity;
    KeyT slot = slots[slot_idx];
    if (slot == key) { found = true; break; }
    if (slot == empty_key) break;
  }
  out[idx] = found;
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

  /// Alias matching the real cuco::static_set_ref API. Sirius calls
  /// `set.contains(key)` inside device kernels
  /// (sirius_dynamic_in_list_filter.cu); without this the ROCm port fails to
  /// compile when SIRIUS_ENABLE_CUCO=ON.
  __host__ __device__ bool contains(KeyT const& key) const {
    return operator()(key);
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
      hash_{},
      last_stream_(stream) {
    (void)alloc; // Allocator accepted for API compat, hipMalloc used
    if (capacity_ == 0) {
      throw std::runtime_error("cuco::static_set: capacity must be greater than 0");
    }
    // Overflow-safe byte count: capacity_ * sizeof(KeyT) can wrap size_t for a
    // caller-supplied capacity near SIZE_MAX, yielding a small hipMalloc while
    // kernels index the full capacity_ → OOB device access.
    if (capacity_ != 0 && sizeof(KeyT) > SIZE_MAX / capacity_) {
      throw std::runtime_error("cuco::static_set: capacity * sizeof(KeyT) overflows size_t");
    }
    std::size_t bytes = capacity_ * sizeof(KeyT);
    if (hipMalloc(&slots_, bytes) != hipSuccess) {
      throw std::runtime_error("cuco::static_set: hipMalloc failed");
    }
    // Fill all slots with the empty key on the caller's stream. Launch enough
    // threads to cover the capacity — previously <<<1,1>>> only filled slot 0.
    std::size_t block = 256;
    std::size_t grid = (capacity_ + block - 1) / block;
    if (grid == 0) grid = 1;
    detail::fill_empty_key_kernel<KeyT><<<grid, block, 0, stream>>>(slots_, empty_key_, capacity_);
    if (hipGetLastError() != hipSuccess) {
      hipFree(slots_);
      slots_ = nullptr;
      throw std::runtime_error("cuco::static_set: fill_empty_key_kernel launch failed");
    }
  }

  __host__ ~static_set() {
    if (slots_) {
      // Sync ALL device work before freeing — hipFree does NOT sync non-default
      // streams, and the set may have been used on multiple streams (insert on
      // one, probe on another). hipDeviceSynchronize guarantees no async kernel
      // is still accessing slots_ when we free it.
      hipDeviceSynchronize();
      hipFree(slots_);
    }
  }

  static_set(static_set const&) = delete;
  static_set& operator=(static_set const&) = delete;

  __host__ __device__ std::size_t capacity() const { return capacity_; }
  __host__ __device__ KeyT* data() { return slots_; }
  __host__ __device__ KeyT const* data() const { return slots_; }

  /// Async insert a range of keys.
  template <typename InputIt>
  __host__ void insert_async(InputIt first, InputIt last, hipStream_t stream) {
    last_stream_ = stream;
    auto n = static_cast<std::size_t>(last - first);
    if (n == 0) return;
    std::size_t block = 256;
    std::size_t grid = (n + block - 1) / block;
    detail::set_insert_kernel<KeyT, Hash, InputIt>
      <<<grid, block, 0, stream>>>(slots_, capacity_, empty_key_, hash_, first, n);
  }

  /// Async probe: write `true`/`false` for each key. Tracks the stream so the
  /// destructor can sync it before freeing slots_ (hipFree does NOT sync
  /// non-default streams). Without this, a probe issued on a non-default
  /// stream that is still in flight when the set is destroyed would read
  /// freed memory.
  template <typename InputIt, typename OutputIt>
  __host__ void contains_async(InputIt first, InputIt last, OutputIt out, hipStream_t stream) {
    last_stream_ = stream;
    auto n = static_cast<std::size_t>(last - first);
    if (n == 0) return;
    std::size_t block = 256;
    std::size_t grid = (n + block - 1) / block;
    detail::set_probe_kernel<KeyT, Hash, InputIt, OutputIt>
      <<<grid, block, 0, stream>>>(slots_, capacity_, empty_key_, hash_, first, n, out);
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
  hipStream_t last_stream_{0};  // tracked for destructor sync
};

}  // namespace cuco
