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
    : upstream_(upstream), reservation_limit_(reservation_limit) {
    // Capture the currently-active HIP device so defragment() targets the
    // correct device's memory pool instead of hardcoding device 0 (wrong on
    // multi-GPU systems). Querying here avoids changing the ctor signature
    // (which external callers use positionally).
    int dev = 0;
    if (hipGetDevice(&dev) == hipSuccess) {
      device_id_ = dev;
    }
  }

  /// Allocate `bytes` (rounded to alignment) on the upstream resource and
  /// charge it against the active reservation for `stream`. On failure, invoke
  /// the attached OOM policy (default rethrows) so Sirius can reschedule.
  void* allocate(rmm::cuda_stream_view stream, std::size_t bytes,
                 std::size_t alignment = 0) {
    // Read the current charge for this stream under the lock; do NOT hold the
    // reference across the (blocking) upstream allocation below, since a
    // concurrent insert on another stream could rehash stream_tracker_ and
    // dangle it.
    std::size_t current;
    {
      std::lock_guard<std::mutex> lk(tracker_mutex_);
      current = stream_tracker_[stream].current_bytes.load(std::memory_order_relaxed);
    }
    // Respect a reservation limit if one was set (0 = unlimited).
    if (reservation_limit_ != 0 && current + bytes > reservation_limit_) {
      std::exception_ptr eptr;
      try {
        throw std::runtime_error(
          "cuCascade: reservation limit exceeded in "
          "reservation_aware_resource_adaptor::allocate");
      } catch (...) {
        eptr = std::current_exception();
      }
      oom_handling_policy::RetryFunc retry =
        [this](std::size_t b, rmm::cuda_stream_view) { return upstream_.allocate(b, 0); };
      // Invoke handle_oom INSIDE the lock: oom_policy_for() previously
      // returned a raw pointer into stream_tracker_ and released the lock, so
      // a concurrent attach/erase could destroy the policy before this call
      // ran (use-after-free). Also use the correct 4-arg signature
      // (requested_bytes, stream, eptr, retry) — the old call passed
      // (bytes, current), which did not match handle_oom's signature.
      {
        std::lock_guard<std::mutex> lk(tracker_mutex_);
        auto it = stream_tracker_.find(stream);
        if (it != stream_tracker_.end() && it->second.oom_policy) {
          it->second.oom_policy->handle_oom(bytes, stream, eptr, retry);
        }
      }
      std::rethrow_exception(eptr);
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
        std::lock_guard<std::mutex> lk(tracker_mutex_);
        host_spilled_[ptr] = {bytes, /*is_uvm=*/true};  // UVM/managed: hipFree
        fprintf(stderr,
          "[sirius] WARNING: GPU OOM — allocated %zu bytes via UVM "
          "(managed memory, total spilled: %zu allocations)\n",
          bytes, host_spilled_.size());
        auto& tracker = stream_tracker_[stream];
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
        std::lock_guard<std::mutex> lk(tracker_mutex_);
        host_spilled_[ptr] = {bytes, /*is_uvm=*/false};  // host-pinned: hipFreeHost
        fprintf(stderr,
          "[sirius] WARNING: GPU OOM — spilled %zu bytes to host memory "
          "(total spilled: %zu allocations)\n",
          bytes, host_spilled_.size());
        auto& tracker = stream_tracker_[stream];
        tracker.current_bytes.fetch_add(bytes, std::memory_order_relaxed);
        std::size_t now = tracker.current_bytes.load(std::memory_order_relaxed);
        std::size_t peak = tracker.peak_bytes.load(std::memory_order_relaxed);
        while (now > peak && !tracker.peak_bytes.compare_exchange_weak(peak, now,
                 std::memory_order_relaxed)) {}
        total_allocated_.fetch_add(bytes, std::memory_order_relaxed);
        return ptr;
      }
      // All tiers failed — invoke OOM policy and rethrow
      // Invoke handle_oom INSIDE the lock (same reason as the
      // reservation-limit path above) and with the correct 4-arg signature.
      {
        std::lock_guard<std::mutex> lk(tracker_mutex_);
        auto it = stream_tracker_.find(stream);
        if (it != stream_tracker_.end() && it->second.oom_policy) {
          oom_handling_policy::RetryFunc retry =
            [this](std::size_t b, rmm::cuda_stream_view) { return upstream_.allocate(b, 0); };
          it->second.oom_policy->handle_oom(bytes, stream, std::current_exception(), retry);
        }
      }
      throw;
    }
    {
      std::lock_guard<std::mutex> lk(tracker_mutex_);
      auto& tracker = stream_tracker_[stream];
      tracker.current_bytes.fetch_add(bytes, std::memory_order_relaxed);
      std::size_t now = tracker.current_bytes.load(std::memory_order_relaxed);
      std::size_t peak = tracker.peak_bytes.load(std::memory_order_relaxed);
      while (now > peak && !tracker.peak_bytes.compare_exchange_weak(peak, now,
               std::memory_order_relaxed)) {}
      total_allocated_.fetch_add(bytes, std::memory_order_relaxed);
    }
    return ptr;
  }

  /// Deallocate `bytes` previously allocated on `stream`, crediting the
  /// reservation tracker. Bytes are clamped at 0 (no underflow).
  /// If the allocation was spilled (OOM fallback), frees it with the function
  /// matching the spill tier recorded at allocate time: hipFree for
  /// UVM/managed memory, hipFreeHost for pinned host memory.
  void deallocate(rmm::cuda_stream_view stream, void* ptr, std::size_t bytes,
                  std::size_t alignment = 0) {
    // Check if this was a spilled allocation (UVM or host-pinned)
    {
      std::lock_guard<std::mutex> lk(tracker_mutex_);
      auto it = host_spilled_.find(ptr);
      if (it != host_spilled_.end()) {
        // Free with the function matching the tier that produced the
        // pointer: hipFree for UVM/managed (hipMallocManaged), hipFreeHost
        // for pinned host memory (hipMallocHost). Guessing (try hipFree then
        // hipFreeHost) is wrong — hipFree on a host-pinned pointer returns an
        // error without freeing, and hipFreeHost on UVM memory is invalid.
        if (it->second.is_uvm) {
          hipFree(ptr);
        } else {
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
    {
      std::lock_guard<std::mutex> lk(tracker_mutex_);
      auto it = stream_tracker_.find(stream);
      if (it != stream_tracker_.end()) {
        std::size_t cur = it->second.current_bytes.load(std::memory_order_relaxed);
        std::size_t sub = cur > bytes ? cur - bytes : 0;
        it->second.current_bytes.store(sub, std::memory_order_relaxed);
      }
    }
  }

  std::size_t get_peak_allocated_bytes(rmm::cuda_stream_view stream) const {
    std::lock_guard<std::mutex> lk(tracker_mutex_);
    auto it = stream_tracker_.find(stream);
    return it != stream_tracker_.end()
             ? it->second.peak_bytes.load(std::memory_order_relaxed)
             : 0;
  }
  void reset_stream_reservation(rmm::cuda_stream_view stream) {
    std::lock_guard<std::mutex> lk(tracker_mutex_);
    auto it = stream_tracker_.find(stream);
    if (it != stream_tracker_.end()) {
      it->second.current_bytes.store(0, std::memory_order_relaxed);
      it->second.peak_bytes.store(0, std::memory_order_relaxed);
    }
  }

  /// Defragment the memory pool. Called between queries when the stream is
  /// idle. Returns unused pages to the OS via hipMemPoolTrimTo, preventing
  /// fragmentation-induced allocation failures on long-running workloads.
  /// Does NOT clear the host-spilled allocation map: those entries are live
  /// allocations (UVM / pinned host) freed individually via deallocate()'s
  /// hipFree/hipFreeHost path, so clearing the map here would orphan them.
  void defragment() {
    std::size_t spilled;
    {
      std::lock_guard<std::mutex> lk(tracker_mutex_);
      spilled = host_spilled_.size();
    }
    // Always return unused pool pages to the OS. hipMemPoolTrimTo only
    // releases free pages from the device memory pool — not live allocations
    // — so it is safe (and useful) to call on the active path, not only when
    // the pool is fully drained.
    hipMemPool_t pool = nullptr;
    if (hipDeviceGetMemPool(&pool, device_id_) == hipSuccess && pool) {
      hipMemPoolTrimTo(pool, 0);
    }
    // If many allocations are spilled to host/UVM, log a warning (indicates
    // sustained memory pressure — the user should consider a larger GPU or
    // smaller batch sizes). Use a fixed count threshold rather than comparing
    // against total_allocated_ (bytes): mixing an allocation count with a byte
    // total is meaningless, and >=100 spilled allocations is a lot regardless
    // of individual size.
    if (spilled > 100) {
      fprintf(stderr,
        "[sirius] WARNING: high memory pressure — %zu allocations spilled to "
        "host/UVM memory. Consider reducing batch size or SIRIUS_MAX_BATCH_BYTES.\n",
        spilled);
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
    std::lock_guard<std::mutex> lk(tracker_mutex_);
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
  int device_id_{0};  // captured in the ctor for defragment()'s hipDeviceGetMemPool
  std::atomic<std::size_t> total_allocated_{0};

  struct stream_state {
    std::atomic<std::size_t> current_bytes{0};
    std::atomic<std::size_t> peak_bytes{0};
    std::unique_ptr<reservation> reservation;
    std::unique_ptr<reservation_limit_policy> limit_policy;
    std::unique_ptr<oom_handling_policy> oom_policy;
  };
  std::unordered_map<rmm::cuda_stream_view, stream_state> stream_tracker_;

  // tracker_mutex_ guards ALL access to both stream_tracker_ and
  // host_spilled_ (stream_tracker_ was previously unlocked — a data race,
  // since a concurrent insert could rehash the map and dangle the
  // references/iterators held across allocate/deallocate). mutable so the
  // const get_peak_allocated_bytes() can lock it.
  mutable std::mutex tracker_mutex_;

  // Host-spilled allocations (OOM fallback): tracks ptr -> {bytes, tier} so
  // deallocate knows the correct free function: hipFree for UVM/managed
  // (hipMallocManaged), hipFreeHost for pinned host memory (hipMallocHost).
  struct spilled_alloc {
    std::size_t bytes;
    bool is_uvm;  // true = UVM/managed (hipFree), false = host-pinned (hipFreeHost)
  };
  std::unordered_map<void*, spilled_alloc> host_spilled_;
};

}  // namespace cucascade::memory
