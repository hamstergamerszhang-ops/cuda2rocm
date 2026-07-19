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
    // Respect a reservation limit if one was set (0 = unlimited). Overflow-
    // safe: `current + bytes > reservation_limit_` can wrap size_t for a huge
    // bytes, falsely passing the check and admitting an over-limit reservation.
    if (reservation_limit_ != 0 &&
        (bytes > reservation_limit_ || current > reservation_limit_ - bytes)) {
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
      // Invoke handle_oom INSIDE the lock (a raw pointer into stream_tracker_
      // would dangle if a concurrent attach/erase rehashed the map) and USE
      // the returned pointer: a recovering policy (e.g.
      // defragmenter_oom_policy) defrags and retries upstream_.allocate,
      // returning the recovered pointer. Discarding it + unconditionally
      // rethrowing (the old code) leaked the recovered allocation and
      // defeated the retry mechanism for any non-throwing policy.
      void* recovered = nullptr;
      {
        std::lock_guard<std::mutex> lk(tracker_mutex_);
        auto it = stream_tracker_.find(stream);
        if (it != stream_tracker_.end() && it->second.oom_policy) {
          recovered = it->second.oom_policy->handle_oom(bytes, stream, eptr, retry);
        }
      }
      if (recovered) {
        // The policy recovered a normal upstream device allocation via retry.
        // Charge the per-stream tracker + global total (no spill entry: it is
        // not a host/UVM fallback) and return it.
        charge_stream(stream, bytes);
        return recovered;
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
      // All tiers failed — invoke OOM policy. As in the reservation-limit path,
      // USE the returned pointer if the policy recovers (e.g. defragmenter
      // defrags then retries upstream_.allocate). The old code discarded the
      // return and unconditionally rethrew, leaking any recovered allocation
      // and defeating the retry mechanism for non-throwing policies.
      {
        void* recovered = nullptr;
        {
          std::lock_guard<std::mutex> lk(tracker_mutex_);
          auto it = stream_tracker_.find(stream);
          if (it != stream_tracker_.end() && it->second.oom_policy) {
            oom_handling_policy::RetryFunc retry =
              [this](std::size_t b, rmm::cuda_stream_view) { return upstream_.allocate(b, 0); };
            recovered =
              it->second.oom_policy->handle_oom(bytes, stream, std::current_exception(), retry);
          }
        }
        if (recovered) {
          // Policy recovered a normal upstream device allocation. Charge it
          // (no spill entry) and return.
          charge_stream(stream, bytes);
          return recovered;
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
          // C9: atomic subtract clamped at 0 — the old load/sub/store was a
          // read-modify-write race that lost updates under concurrent
          // allocate (fetch_add) or concurrent dealloc on the same stream.
          clamped_sub(tracker_it->second.current_bytes, bytes);
        }
        // Gap-2 fix: use clamped_sub (not raw fetch_sub) for the global
        // counter too, matching the normal dealloc path. The old spilled
        // path used fetch_sub unconditionally — if there was any bookkeeping
        // asymmetry (e.g. a double dealloc, or a dealloc without a matching
        // spilled alloc charge), it would wrap total_allocated_ to ~SIZE_MAX,
        // permanently breaking the memory accounting the defragment warning
        // and OOM backpressure rely on.
        clamped_sub(total_allocated_, bytes);
        return;
      }
    }
    // Normal device allocation — delegate to upstream
    upstream_.deallocate(ptr, bytes, alignment);
    {
      std::lock_guard<std::mutex> lk(tracker_mutex_);
      auto it = stream_tracker_.find(stream);
      if (it != stream_tracker_.end()) {
        // C9: same clamped atomic subtract as the spilled path.
        clamped_sub(it->second.current_bytes, bytes);
      }
      // C10: the normal (non-spilled) deallocate path previously did NOT
      // decrement total_allocated_, while allocate always increments it — so
      // the global counter grew monotonically across every normal
      // alloc/dealloc cycle, breaking memory accounting (e.g. defragment's
      // pressure warning fired forever after the first freed allocation).
      clamped_sub(total_allocated_, bytes);
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
  /// Charge `bytes` to the per-stream tracker (current + peak) and the global
  /// total. Used by the OOM-recovery paths that obtain a pointer via the
  /// policy's retry (a normal upstream device allocation, not a spill).
  void charge_stream(rmm::cuda_stream_view stream, std::size_t bytes) {
    std::lock_guard<std::mutex> lk(tracker_mutex_);
    auto& tracker = stream_tracker_[stream];
    tracker.current_bytes.fetch_add(bytes, std::memory_order_relaxed);
    std::size_t now = tracker.current_bytes.load(std::memory_order_relaxed);
    std::size_t peak = tracker.peak_bytes.load(std::memory_order_relaxed);
    while (now > peak && !tracker.peak_bytes.compare_exchange_weak(peak, now,
             std::memory_order_relaxed)) {}
    total_allocated_.fetch_add(bytes, std::memory_order_relaxed);
  }

  /// Subtract `bytes` from an atomic counter, clamped at 0. CAS loop — no
  /// read-modify-write race (the old load/sub/store lost updates under
  /// concurrent fetch_add / concurrent dealloc on the same stream) and no
  /// underflow (a bookkeeping asymmetry can't wrap the counter to ~SIZE_MAX).
  static void clamped_sub(std::atomic<std::size_t>& counter, std::size_t bytes) {
    std::size_t cur = counter.load(std::memory_order_relaxed);
    while (cur > 0) {
      std::size_t next = cur > bytes ? cur - bytes : 0;
      if (counter.compare_exchange_weak(cur, next, std::memory_order_relaxed)) {
        break;
      }
      // cur reloaded by the failed CAS; retry.
    }
  }

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
