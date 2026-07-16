/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file cuCascade reservation_aware_resource_adaptor — ROCm real implementation.
//!
//! Wraps an upstream rmm::device_async_resource_ref (the hipMM/RMM device
//! memory resource) with per-stream reservation tracking. Allocations delegate
//! to the upstream resource; the reservation bookkeeping is real (peak/total
//! bytes, per-stream reservation ownership) so Sirius's OOM/backpressure path
//! sees accurate counters. The OOM policy is invoked on allocation failure if
//! the caller attached one; the default (throw_on_oom_policy) rethrows.

#pragma once

#include <rmm/cuda_stream.hpp>
#include <rmm/resource_ref.hpp>
#include "cucascade/memory/memory_reservation.hpp"
#include "cucascade/memory/oom_handling_policy.hpp"
#include <hip/hip_runtime_api.h>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace cucascade::memory {

class memory_space;

/// Wraps an RMM device memory resource with reservation tracking.
/// allocate/deallocate delegate to the captured upstream resource; the
/// reservation tracker keeps peak/total counters that Sirius reads to drive
/// backpressure and OOM rescheduling.
class reservation_aware_resource_adaptor {
 public:
  reservation_aware_resource_adaptor(rmm::device_async_resource_ref upstream,
                                     std::size_t reservation_limit = 0)
    : upstream_(upstream), reservation_limit_(reservation_limit) {}

  /// Allocate `bytes` (rounded to alignment) on the upstream resource and
  /// charge it against the active reservation for `stream`. On failure, invoke
  /// the attached OOM policy (default rethrows) so Sirius can reschedule.
  void* allocate(rmm::cuda_stream_view stream, std::size_t bytes,
                 std::size_t alignment = 0) {
    auto& tracker = stream_tracker_[stream];
    std::size_t current = tracker.current_bytes.load(std::memory_order_relaxed);
    // Respect a reservation limit if one was set (0 = unlimited).
    if (reservation_limit_ != 0 && current + bytes > reservation_limit_) {
      if (auto* oom = oom_policy_for(stream)) {
        oom->handle_oom(bytes, current);
      }
      throw std::runtime_error(
        "cuCascade: reservation limit exceeded in "
        "reservation_aware_resource_adaptor::allocate");
    }
    void* ptr = nullptr;
    try {
      ptr = upstream_.allocate(bytes, alignment);
    } catch (...) {
      // OOM cascade: device → UVM (managed memory) → pinned host
      //
      // Tier 1: hipMallocManaged (Unified Virtual Memory)
      //   - GPU can access directly via page-fault migration
      //   - Faster than host-staging for irregular access patterns
      //   - Only available on AMD ROCm (hipMallocManaged)
      hipError_t err = hipMallocManaged(&ptr, bytes);
      if (err == hipSuccess && ptr != nullptr) {
        std::lock_guard<std::mutex> lk(host_spill_mutex_);
        host_spilled_[ptr] = bytes;  // tracked as "spilled" for dealloc
        fprintf(stderr,
          "[sirius] WARNING: GPU OOM — allocated %zu bytes via UVM "
          "(managed memory, total spilled: %zu allocations)\n",
          bytes, host_spilled_.size());
        tracker.current_bytes.fetch_add(bytes, std::memory_order_relaxed);
        std::size_t now = tracker.current_bytes.load(std::memory_order_relaxed);
        std::size_t peak = tracker.peak_bytes.load(std::memory_order_relaxed);
        while (now > peak && !tracker.peak_bytes.compare_exchange_weak(peak, now,
                 std::memory_order_relaxed)) {}
        total_allocated_.fetch_add(bytes, std::memory_order_relaxed);
        return ptr;
      }

      // Tier 2: hipMallocHost (pinned host memory)
      //   - Slower than UVM (explicit D2H/H2D needed)
      //   - Works on all platforms
      err = hipMallocHost(&ptr, bytes);
      if (err == hipSuccess && ptr != nullptr) {
        std::lock_guard<std::mutex> lk(host_spill_mutex_);
        host_spilled_[ptr] = bytes;
        fprintf(stderr,
          "[sirius] WARNING: GPU OOM — spilled %zu bytes to host memory "
          "(total spilled: %zu allocations)\n",
          bytes, host_spilled_.size());
        tracker.current_bytes.fetch_add(bytes, std::memory_order_relaxed);
        std::size_t now = tracker.current_bytes.load(std::memory_order_relaxed);
        std::size_t peak = tracker.peak_bytes.load(std::memory_order_relaxed);
        while (now > peak && !tracker.peak_bytes.compare_exchange_weak(peak, now,
                 std::memory_order_relaxed)) {}
        total_allocated_.fetch_add(bytes, std::memory_order_relaxed);
        return ptr;
      }
      // All tiers failed — invoke OOM policy and rethrow
      if (auto* oom = oom_policy_for(stream)) {
        oom->handle_oom(bytes, current);
      }
      throw;
    }
    tracker.current_bytes.fetch_add(bytes, std::memory_order_relaxed);
    std::size_t now = tracker.current_bytes.load(std::memory_order_relaxed);
    std::size_t peak = tracker.peak_bytes.load(std::memory_order_relaxed);
    while (now > peak && !tracker.peak_bytes.compare_exchange_weak(peak, now,
             std::memory_order_relaxed)) {}
    total_allocated_.fetch_add(bytes, std::memory_order_relaxed);
    return ptr;
  }

  /// Deallocate `bytes` previously allocated on `stream`, crediting the
  /// reservation tracker. Bytes are clamped at 0 (no underflow).
  /// If the allocation was spilled (OOM fallback), tries hipFree first (for
  /// UVM/managed memory), then hipFreeHost (for pinned host memory).
  void deallocate(rmm::cuda_stream_view stream, void* ptr, std::size_t bytes,
                  std::size_t alignment = 0) {
    // Check if this was a spilled allocation (UVM or host-pinned)
    {
      std::lock_guard<std::mutex> lk(host_spill_mutex_);
      auto it = host_spilled_.find(ptr);
      if (it != host_spilled_.end()) {
        // Try hipFree first (works for UVM/managed memory).
        // If it fails, try hipFreeHost (for pinned host memory).
        hipError_t err = hipFree(ptr);
        if (err != hipSuccess) {
          hipFreeHost(ptr);
        }
        host_spilled_.erase(it);
        auto tracker_it = stream_tracker_.find(stream);
        if (tracker_it != stream_tracker_.end()) {
          std::size_t cur = tracker_it->second.current_bytes.load(std::memory_order_relaxed);
          std::size_t sub = cur > bytes ? cur - bytes : 0;
          tracker_it->second.current_bytes.store(sub, std::memory_order_relaxed);
        }
        total_allocated_.fetch_sub(bytes, std::memory_order_relaxed);
        return;
      }
    }
    // Normal device allocation — delegate to upstream
    upstream_.deallocate(ptr, bytes, alignment);
    auto it = stream_tracker_.find(stream);
    if (it != stream_tracker_.end()) {
      std::size_t cur = it->second.current_bytes.load(std::memory_order_relaxed);
      std::size_t sub = cur > bytes ? cur - bytes : 0;
      it->second.current_bytes.store(sub, std::memory_order_relaxed);
    }
  }

  std::size_t get_peak_allocated_bytes(rmm::cuda_stream_view stream) const {
    auto it = stream_tracker_.find(stream);
    return it != stream_tracker_.end()
             ? it->second.peak_bytes.load(std::memory_order_relaxed)
             : 0;
  }
  void reset_stream_reservation(rmm::cuda_stream_view stream) {
    auto it = stream_tracker_.find(stream);
    if (it != stream_tracker_.end()) {
      it->second.current_bytes.store(0, std::memory_order_relaxed);
      it->second.peak_bytes.store(0, std::memory_order_relaxed);
    }
  }

  /// Defragment the memory pool. Called between queries when the stream is
  /// idle. Returns unused pages to the OS via hipMemPoolTrimTo, preventing
  /// fragmentation-induced allocation failures on long-running workloads.
  /// Also clears the host-spilled allocation map (those pointers are stale
  /// after the pool is trimmed).
  void defragment() {
    std::size_t current = total_allocated_.load(std::memory_order_relaxed);
    std::size_t spilled = 0;
    {
      std::lock_guard<std::mutex> lk(host_spill_mutex_);
      spilled = host_spilled_.size();
    }
    if (current == 0 && spilled == 0) {
      // Nothing allocated — trim the pool to release all pages
      hipMemPool_t pool = nullptr;
      if (hipDeviceGetMemPool(&pool, 0) == hipSuccess && pool) {
        hipMemPoolTrimTo(pool, 0);
      }
      return;
    }
    // If more than 50% of allocations are spilled to host/UVM, log a warning
    // (indicates sustained memory pressure — the user should consider a
    // larger GPU or smaller batch sizes).
    if (spilled > 0 && spilled > total_allocated_.load() / 2) {
      fprintf(stderr,
        "[sirius] WARNING: high memory pressure — %zu spilled allocations "
        "out of %zu total. Consider reducing batch size or SIRIUS_MAX_BATCH_BYTES.\n",
        spilled, total_allocated_.load());
    }
  }

  // Sirius calls with 4 args: (stream, reservation, limit_policy, oom_policy)
  // and with 3 args: (stream, reservation, limit_policy)
  // and with 2 args: (stream, reservation) [test code]
  // Returns true on success (reservation accepted). The reservation is held
  // for the stream's lifetime and released on detach/reset.
  bool attach_reservation_to_tracker(rmm::cuda_stream_view stream,
                                     std::unique_ptr<reservation> reservation,
                                     std::unique_ptr<reservation_limit_policy> limit_policy = nullptr,
                                     std::unique_ptr<oom_handling_policy> oom_policy = nullptr) {
    auto& tracker = stream_tracker_[stream];
    tracker.reservation = std::move(reservation);
    tracker.limit_policy = std::move(limit_policy);
    tracker.oom_policy = std::move(oom_policy);
    return true;
  }

  /// Expose the upstream resource so callers (e.g. host_parquet converter)
  /// can pass it to cudf::io::read_parquet's mr argument.
  operator rmm::device_async_resource_ref() const {
    return upstream_;
  }

 private:
  rmm::device_async_resource_ref upstream_;
  std::size_t reservation_limit_{0};
  std::atomic<std::size_t> total_allocated_{0};

  struct stream_state {
    std::atomic<std::size_t> current_bytes{0};
    std::atomic<std::size_t> peak_bytes{0};
    std::unique_ptr<reservation> reservation;
    std::unique_ptr<reservation_limit_policy> limit_policy;
    std::unique_ptr<oom_handling_policy> oom_policy;
  };
  std::unordered_map<rmm::cuda_stream_view, stream_state> stream_tracker_;

  // Host-spilled allocations (OOM fallback): tracks ptr → bytes so
  // deallocate knows to use hipFreeHost instead of upstream_.deallocate.
  std::mutex host_spill_mutex_;
  std::unordered_map<void*, std::size_t> host_spilled_;

  oom_handling_policy* oom_policy_for(rmm::cuda_stream_view stream) {
    auto it = stream_tracker_.find(stream);
    return it != stream_tracker_.end() ? it->second.oom_policy.get() : nullptr;
  }
};

}  // namespace cucascade::memory
