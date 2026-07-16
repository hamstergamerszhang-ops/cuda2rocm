/*
 * Copyright 2026, rocm-cuda-compat contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file cuCascade memory_space — ROCm real implementation.
//! The central type: ties together a tier, device, stream pool, and allocator.
//! Resource accessors return real pointers to stored resource objects.

#pragma once

#include "cucascade/memory/common.hpp"
#include "cucascade/memory/memory_reservation.hpp"
#include "cucascade/memory/stream_pool.hpp"
#include "cucascade/memory/reservation_aware_resource_adaptor.hpp"
#include "cucascade/memory/fixed_size_host_memory_resource.hpp"
#include <rmm/cuda_stream.hpp>
#include <rmm/resource_ref.hpp>
#include <algorithm>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>

namespace cucascade::memory {

/// A memory space represents a tier (GPU/HOST/DISK) on a specific device,
/// with an associated allocator and stream pool.
class memory_space {
 public:
  memory_space(Tier tier, int32_t device_id,
               rmm::device_async_resource_ref allocator,
               std::shared_ptr<exclusive_stream_pool> streams)
    : tier_(tier), device_id_(device_id), allocator_(allocator), streams_(std::move(streams)) {}

  memory_space_id get_id() const { return {tier_, device_id_}; }
  Tier get_tier() const { return tier_; }
  int32_t get_device_id() const { return device_id_; }
  rmm::device_async_resource_ref get_default_allocator() const { return allocator_; }

  rmm::cuda_stream_view acquire_stream() {
    if (!streams_) throw std::runtime_error("cuCascade: no stream pool");
    return streams_->acquire_stream(stream_acquire_policy::GROW);
  }
  rmm::cuda_stream_view acquire_stream() const {
    if (streams_) return streams_->acquire_stream(stream_acquire_policy::GROW);
    return rmm::cuda_stream_per_thread;
  }

  bool should_downgrade_memory() const { return false; }

  std::string to_string() const {
    return "cuCascade memory_space(tier=" + std::to_string(static_cast<int>(tier_)) +
           ",device=" + std::to_string(device_id_) + ")";
  }

  bool operator==(memory_space const& other) const { return get_id() == other.get_id(); }
  bool operator!=(memory_space const& other) const { return !(*this == other); }

  /// Template: get memory resource as a specific type.
  /// Returns a pointer to the stored resource if T matches, nullptr otherwise.
  /// Sirius calls this with T = reservation_aware_resource_adaptor (GPU tier)
  /// or T = fixed_size_host_memory_resource (HOST tier).
  template <typename T>
  T* get_memory_resource_as() {
    return get_resource_ptr<T>();
  }
  template <typename T>
  T const* get_memory_resource_as() const {
    return get_resource_ptr<T>();
  }

  /// Template: get memory resource for a specific tier.
  template <Tier TIER>
  typename tier_memory_resource_trait<TIER>::type* get_memory_resource_of() {
    return get_resource_ptr<typename tier_memory_resource_trait<TIER>::type>();
  }
  template <Tier TIER>
  typename tier_memory_resource_trait<TIER>::type const* get_memory_resource_of() const {
    return get_resource_ptr<typename tier_memory_resource_trait<TIER>::type>();
  }

  // --- Memory tracking (real: tracks allocated/reserved bytes) ---
  std::size_t get_max_memory() const { return max_memory_; }
  // Guard against unsigned underflow: if max_memory_ was never set (defaults
  // to 0) or reserved exceeds max (shouldn't happen, but defensive), return 0
  // instead of wrapping to a huge value. Without this, an uninitialized
  // max_memory_=0 makes get_available_memory() return ~0ULL, so every
  // reservation succeeds and the OOM/backpressure path is defeated.
  std::size_t get_available_memory() const {
    std::size_t reserved = reserved_memory_.load(std::memory_order_relaxed);
    return max_memory_ > reserved ? max_memory_ - reserved : 0;
  }
  std::size_t get_total_reserved_memory() const { return reserved_memory_.load(std::memory_order_relaxed); }

  std::unique_ptr<reservation> make_reservation(
    std::size_t bytes,
    reservation::release_callback on_release = {}) {
    if (bytes > get_available_memory()) {
      throw std::runtime_error("cuCascade: insufficient memory for reservation");
    }
    reserved_memory_.fetch_add(bytes, std::memory_order_relaxed);
    auto arena = std::make_unique<simple_arena>(bytes);
    auto space_release = [this, bytes, cb = std::move(on_release)]() mutable {
      release_reservation(bytes);
      if (cb) cb();
    };
    return reservation::create(*this, std::move(arena), std::move(space_release));
  }
  std::unique_ptr<reservation> make_reservation_or_null(
    std::size_t bytes,
    reservation::release_callback on_release = {}) {
    if (bytes > get_available_memory()) return nullptr;
    reserved_memory_.fetch_add(bytes, std::memory_order_relaxed);
    auto arena = std::make_unique<simple_arena>(bytes);
    auto space_release = [this, bytes, cb = std::move(on_release)]() mutable {
      release_reservation(bytes);
      if (cb) cb();
    };
    return reservation::create(*this, std::move(arena), std::move(space_release));
  }
  std::unique_ptr<reservation> make_reservation_upto(
    std::size_t bytes,
    reservation::release_callback on_release = {}) {
    std::size_t actual = std::min(bytes, get_available_memory());
    if (actual == 0) return nullptr;
    reserved_memory_.fetch_add(actual, std::memory_order_relaxed);
    auto arena = std::make_unique<simple_arena>(actual);
    auto space_release = [this, actual, cb = std::move(on_release)]() mutable {
      release_reservation(actual);
      if (cb) cb();
    };
    return reservation::create(*this, std::move(arena), std::move(space_release));
  }

  void release_reservation(std::size_t bytes) {
    // CAS loop preserves the original no-underflow guard (only subtract when
    // reserved >= bytes) while making the read-modify-write atomic.
    std::size_t cur = reserved_memory_.load(std::memory_order_relaxed);
    while (cur >= bytes &&
           !reserved_memory_.compare_exchange_weak(cur, cur - bytes,
                                                    std::memory_order_relaxed)) {}
  }

  // --- Setters (used by memory_reservation_manager during construction) ---
  void set_max_memory(std::size_t bytes) { max_memory_ = bytes; }
  void set_gpu_resource(std::unique_ptr<reservation_aware_resource_adaptor> res) {
    gpu_resource_ = std::move(res);
  }
  void set_host_resource(std::unique_ptr<fixed_size_host_memory_resource> res) {
    host_resource_ = std::move(res);
  }

 private:
  Tier tier_;
  int32_t device_id_;
  rmm::device_async_resource_ref allocator_;
  std::shared_ptr<exclusive_stream_pool> streams_;
  std::size_t max_memory_{0};
  std::atomic<std::size_t> reserved_memory_{0};

  // Stored resources — one per tier. get_memory_resource_as<T>() returns
  // the matching one (or nullptr if T doesn't match any stored resource).
  std::unique_ptr<reservation_aware_resource_adaptor> gpu_resource_;
  std::unique_ptr<fixed_size_host_memory_resource> host_resource_;

  /// Helper: return pointer to stored resource if T matches.
  template <typename T>
  T* get_resource_ptr() {
    if constexpr (std::is_same_v<T, reservation_aware_resource_adaptor>) {
      return gpu_resource_.get();
    } else if constexpr (std::is_same_v<T, fixed_size_host_memory_resource>) {
      return host_resource_.get();
    } else {
      return nullptr;
    }
  }
  template <typename T>
  T const* get_resource_ptr() const {
    if constexpr (std::is_same_v<T, reservation_aware_resource_adaptor>) {
      return gpu_resource_.get();
    } else if constexpr (std::is_same_v<T, fixed_size_host_memory_resource>) {
      return host_resource_.get();
    } else {
      return nullptr;
    }
  }

  /// Simple arena: tracks a byte count.
  class simple_arena : public reserved_arena {
   public:
    explicit simple_arena(std::size_t n) : bytes_(n) {}
    std::size_t size() const override { return bytes_; }
    void grow_by(std::size_t n) override { bytes_ += n; }
    void shrink_to_fit() override {}
   private:
    std::size_t bytes_{0};
  };
};

}  // namespace cucascade::memory
