/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade fixed_size_host_memory_resource — ROCm real implementation.
//! Pinned host RAM pool backed by hipMallocHost / hipFreeHost.
//! multiple_blocks_allocation is NESTED inside the class (Sirius references
//! it as fixed_size_host_memory_resource::multiple_blocks_allocation).
//!
//! Accounting model (ports the real cuCascade nested single-counter design):
//! There is ONE committed-bytes counter, `total_allocated_`, bounded by
//! `mem_limit_`. A reservation is a PRE-CHARGE to that counter
//! (`reserve()` -> `do_reserve()` -> atomic CAS `try_add`). An allocation
//! made under a reservation charges the counter ONLY for the overflow beyond
//! the reservation size (0 if it fits inside, `post - reservation_size` if it
//! straddles the boundary, the full amount once the reservation is already
//! consumed). A reservation release credits only the UNCONSUMED portion
//! (`arena_size - allocated_under_it`). A deallocation credits the overflow
//! that was charged at allocate time, recomputed symmetrically from the
//! per-reservation tracker. Each byte is therefore counted exactly once —
//! there is no `allocated + reserved` sum to double-count the overlap, and
//! `get_available_memory()` (which reads the single counter) agrees with the
//! `reserve()` admission check. Mirrors
//! sirius/cucascade/src/memory/fixed_size_host_memory_resource.cpp.

#pragma once
#include "cucascade/memory/common.hpp"
#include "cucascade/memory/memory_reservation.hpp"
#include "cucascade/memory/notification_channel.hpp"
#include <rmm/cuda_stream.hpp>
#include <rmm/resource_ref.hpp>
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cucascade::memory {

/// Fixed-size-block host memory resource (pinned host RAM pool).
/// Allocations are backed by hipMallocHost (page-locked host memory) and freed
/// with hipFreeHost. Block granularity is `block_size_`; requests are rounded
/// up to the next block. Reservation tracking is real so Sirius's host-staging
/// backpressure path sees accurate counters.
class fixed_size_host_memory_resource {
 public:
  // Sirius accesses these as fixed_size_host_memory_resource::default_pool_size etc.
  static constexpr std::size_t default_pool_size = ::cucascade::memory::default_pool_size;
  static constexpr std::size_t default_block_size = ::cucascade::memory::default_block_size;
  static constexpr std::size_t default_initial_number_pools = ::cucascade::memory::default_initial_number_pools;

  /// A memory block: a pointer + its size. Sirius calls .data() and uses
  /// the pointer for arithmetic/memcpy. It also implicitly converts to void*
  /// for code that treats blocks as raw pointers (memcpy etc.).
  struct block {
    void* ptr{nullptr};
    std::size_t bytes{0};

    block() = default;
    block(void* p, std::size_t n) : ptr(p), bytes(n) {}

    /// Sirius calls .data() on block elements.
    void* data() const { return ptr; }
    std::size_t size() const { return bytes; }

    /// Implicit conversion to void* — Sirius uses blocks[i] in memcpy.
    operator void*() const { return ptr; }
  };

  /// Multiple-block allocation from this resource. NESTED type — Sirius
  /// references it as fixed_size_host_memory_resource::multiple_blocks_allocation.
  struct multiple_blocks_allocation {
    /// Factory: creates from a buffer, memory resource, and reservation.
    /// Sirius calls multiple_blocks_allocation::create(buf, mr, reservation*).
    static std::shared_ptr<multiple_blocks_allocation> create(
      std::shared_ptr<void> /*buffer*/,
      rmm::device_async_resource_ref /*mr*/,
      reservation* /*res*/) {
      return std::make_shared<multiple_blocks_allocation>();
    }

    static std::unique_ptr<multiple_blocks_allocation> create_unique() {
      return std::make_unique<multiple_blocks_allocation>();
    }

    bool empty() const { return blocks_.empty(); }
    std::size_t size_bytes() const { return total_bytes_; }
    std::size_t size() const { return blocks_.size(); }
    void release_blocks() { blocks_.clear(); }

    block& operator[](std::size_t i) {
      return i < blocks_.size() ? blocks_[i] : throw std::out_of_range("block index");
    }
    block const& operator[](std::size_t i) const {
      return i < blocks_.size() ? blocks_[i] : throw std::out_of_range("block index");
    }
    block const& at(std::size_t i) const { return operator[](i); }

    /// Accessor for the blocks vector. Sirius calls .get_blocks().size()
    /// and .get_blocks()[i] (as void* via block's implicit conversion).
    std::vector<block> const& get_blocks() const { return blocks_; }
    std::vector<block>& get_blocks() { return blocks_; }

    std::size_t block_size() const { return block_size_; }
    void grow_by(std::size_t n) { total_bytes_ += n; }
    void trim_to(std::size_t n) { total_bytes_ = n; }

    std::vector<block> blocks_;
    std::size_t block_size_{default_block_size};
    std::size_t total_bytes_{0};
    /// The reservation (arena) this allocation was charged against, or null
    /// for an unmanaged allocation. deallocate_multiple_blocks uses it to
    /// recompute the symmetric overflow credit against the per-reservation
    /// tracker (or, if the reservation was already released, to credit the
    /// full allocation back to the single counter).
    reserved_arena* arena_{nullptr};
  };

  using fixed_multiple_blocks_allocation = std::unique_ptr<multiple_blocks_allocation>;

  fixed_size_host_memory_resource(int32_t device_id,
                                  rmm::device_async_resource_ref upstream,
                                  std::size_t mem_limit = 0,
                                  std::size_t /*capacity*/ = 0,
                                  std::size_t block_size = default_block_size,
                                  std::size_t /*pool_size*/ = default_pool_size,
                                  std::size_t /*initial_pools*/ = default_initial_number_pools)
    : device_id_(device_id),
      upstream_(upstream),
      mem_limit_(mem_limit),
      block_size_(block_size == 0 ? default_block_size : block_size) {}

  std::size_t get_block_size() const { return block_size_; }
  std::size_t get_available_memory() const {
    std::size_t used = total_allocated_.load(std::memory_order_relaxed);
    return mem_limit_ > used ? mem_limit_ - used : 0;
  }
  std::size_t get_free_blocks() const {
    // Number of whole blocks that still fit in the remaining budget.
    std::size_t avail = get_available_memory();
    return avail / block_size_;
  }
  std::size_t get_total_blocks() const {
    // Note: only meaningful when allocations come exclusively through
    // allocate_multiple_blocks (which rounds up to block_size_). The single
    // allocate() path charges raw bytes, so this division undercounts in that
    // mixed case — matching the real cuCascade, whose allocate() throws and
    // is therefore never mixed in.
    return total_allocated_.load(std::memory_order_relaxed) / block_size_;
  }
  std::size_t get_total_allocated_bytes() const {
    return total_allocated_.load(std::memory_order_relaxed);
  }
  std::size_t get_total_reserved_bytes() const {
    // Sum live reservation (arena) sizes. This is the same quantity the
    // admission check bounds (reservations are a subset of the single
    // counter), so the accessor and the check now agree.
    std::lock_guard<std::mutex> lk(mutex_);
    std::size_t total = 0;
    for (const auto& [arena, tracker] : active_reservations_) {
      total += arena ? arena->size() : 0;
    }
    return total;
  }
  std::size_t get_active_reservation_count() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return active_reservations_.size();
  }
  std::size_t get_peak_total_allocated_bytes() const {
    return peak_allocated_.load(std::memory_order_relaxed);
  }

  /// Reserve `bytes` of host capacity. Returns a reservation handle whose
  /// release callback credits the UNCONSUMED portion back to the single
  /// counter. Returns nullptr if the capacity is full (mirrors the real
  /// cuCascade, so a delegating memory_space can fall back); throws only for
  /// an oversized single reservation.
  std::unique_ptr<reservation> reserve(std::size_t bytes,
                                       notification_channel* /*notifier*/ = nullptr) {
    bytes = align_up(bytes, block_size_);
    if (mem_limit_ != 0 && bytes > mem_limit_) {
      throw std::runtime_error(
        "cuCascade: fixed_size_host_memory_resource: reservation size " +
        std::to_string(bytes) + " exceeds memory limit " + std::to_string(mem_limit_));
    }
    // Single-counter atomic check-and-charge (CAS). No read-then-fetch_add
    // race: the CAS only commits `bytes` if `current + bytes <= mem_limit_`,
    // so two concurrent reserves cannot both pass and oversubscribe. No
    // `allocated + reserved` sum: reserved bytes are charged into the SAME
    // counter allocations use, so the overlap is structurally impossible.
    if (!do_reserve(bytes, mem_limit_)) {
      return nullptr;
    }
    auto arena = std::make_unique<reserved_arena_bytes>(bytes);
    reserved_arena* arena_raw = arena.get();
    {
      std::lock_guard<std::mutex> lk(mutex_);
      // try_emplace constructs the tracker IN PLACE in the map node — no move.
      // allocation_tracker holds a std::atomic (non-copyable, non-movable),
      // so emplace(key, allocation_tracker{}) would fail (it would need to
      // move the temporary). try_emplace value-initializes the tracker in the
      // node, leaving allocated_bytes == 0.
      active_reservations_.try_emplace(arena_raw);
    }
    auto* self = this;
    auto on_release = [self, arena_raw]() {
      self->release_reservation(arena_raw);
    };
    // Host-only reservations are not backed by a memory_space; create the
    // reservation with a null space so tier()/device_id() return HOST/0 and
    // get_memory_space() throws if called.
    return reservation::create(std::move(arena), std::move(on_release));
  }

  /// Allocate `bytes` of pinned host memory, rounded up to block_size_.
  /// If `res` is a live reservation on this resource, the allocation is
  /// charged against it: only the overflow beyond the reservation size is
  /// added to the single counter (0 if it fits inside, the straddling excess
  /// if partial, the full amount once the reservation is consumed).
  fixed_multiple_blocks_allocation allocate_multiple_blocks(std::size_t bytes,
                                                            reservation* res = nullptr) {
    if (bytes == 0) {
      return std::make_unique<multiple_blocks_allocation>();
    }
    bytes = align_up(bytes, block_size_);
    std::size_t n_blocks = bytes / block_size_;
    std::size_t alloc_bytes = n_blocks * block_size_;

    reserved_arena* arena = res ? res->arena() : nullptr;
    allocation_tracker* tracker = nullptr;
    std::size_t upstream_tracked_bytes = alloc_bytes;  // default: full charge

    // Hold the mutex across tracker lookup + charge + counter CAS + live
    // registration so a concurrent release_reservation cannot erase (and
    // thus destroy) this arena's tracker between the charge and the
    // symmetric credit. Mirrors the real cuCascade, which holds _mutex
    // across the equivalent span. hipMallocHost runs under the lock too —
    // the port has no free-block pool to defer to, matching sirius's
    // expand_pool-under-lock slow path.
    std::lock_guard<std::mutex> lk(mutex_);
    if (arena) {
      auto it = active_reservations_.find(arena);
      if (it == active_reservations_.end()) {
        throw std::runtime_error(
          "cuCascade: fixed_size_host_memory_resource: reservation has been freed already");
      }
      tracker = std::addressof(it->second);
      auto reservation_size = static_cast<int64_t>(arena->size());
      int64_t pre_allocation_size =
        tracker->allocated_bytes.fetch_add(static_cast<int64_t>(alloc_bytes),
                                           std::memory_order_relaxed);
      int64_t post_allocation_size =
        pre_allocation_size + static_cast<int64_t>(alloc_bytes);
      if (post_allocation_size <= reservation_size) {
        upstream_tracked_bytes = 0;                          // fully inside reservation
      } else if (pre_allocation_size < reservation_size) {
        upstream_tracked_bytes = static_cast<std::size_t>(  // only the straddling overflow
          post_allocation_size - reservation_size);
      }
      // else: reservation already consumed -> full charge (default).
    }

    if (!do_reserve(upstream_tracked_bytes, mem_limit_)) {
      // Roll back the per-reservation tracker charge before failing.
      if (tracker) {
        tracker->allocated_bytes.fetch_sub(static_cast<int64_t>(alloc_bytes),
                                           std::memory_order_relaxed);
      }
      throw std::runtime_error(
        std::string("cuCascade: fixed_size_host_memory_resource OOM: cannot allocate ") +
        std::to_string(alloc_bytes) + " bytes (upstream_tracked=" +
        std::to_string(upstream_tracked_bytes) + ") within memory limit " +
        std::to_string(mem_limit_));
    }

    void* ptr = nullptr;
    hipError_t err = hipMallocHost(&ptr, alloc_bytes);
    if (err != hipSuccess || ptr == nullptr) {
      // Roll back the counter and the tracker charge before rethrowing.
      do_release(upstream_tracked_bytes);
      if (tracker) {
        tracker->allocated_bytes.fetch_sub(static_cast<int64_t>(alloc_bytes),
                                           std::memory_order_relaxed);
      }
      throw std::runtime_error(
        std::string("cuCascade: hipMallocHost failed in "
                    "fixed_size_host_memory_resource::allocate_multiple_blocks: ") +
        hipGetErrorString(err));
    }
    std::memset(ptr, 0, alloc_bytes);
    auto alloc = std::make_unique<multiple_blocks_allocation>();
    alloc->blocks_.reserve(n_blocks);
    for (std::size_t i = 0; i < n_blocks; ++i) {
      alloc->blocks_.emplace_back(
        static_cast<char*>(ptr) + i * block_size_, block_size_);
    }
    alloc->block_size_ = block_size_;
    alloc->total_bytes_ = alloc_bytes;
    alloc->arena_ = arena;
    // Track the raw pointer + size for deallocate_multiple_blocks.
    live_allocs_[reinterpret_cast<uintptr_t>(alloc.get())] = {ptr, alloc_bytes};
    return alloc;
  }

  void deallocate_multiple_blocks(fixed_multiple_blocks_allocation alloc) {
    if (!alloc) return;
    reserved_arena* arena = alloc->arena_;

    void* ptr = nullptr;
    std::size_t tracked_bytes = 0;
    {
      std::lock_guard<std::mutex> lk(mutex_);
      auto it = live_allocs_.find(reinterpret_cast<uintptr_t>(alloc.get()));
      if (it != live_allocs_.end()) {
        ptr = it->second.first;
        tracked_bytes = it->second.second;
        live_allocs_.erase(it);
      }
      if (ptr && arena) {
        auto rit = active_reservations_.find(arena);
        if (rit != active_reservations_.end()) {
          // Reservation still live: credit the symmetric overflow back to
          // the single counter (0 if it was inside the reservation, the
          // straddling excess if partial, the full amount if it overflowed).
          auto& tracker = rit->second;
          auto reservation_size = static_cast<int64_t>(arena->size());
          int64_t pre_reclaimation_size =
            tracker.allocated_bytes.fetch_sub(static_cast<int64_t>(tracked_bytes),
                                              std::memory_order_relaxed);
          int64_t post_reclaimation_size =
            pre_reclaimation_size - static_cast<int64_t>(tracked_bytes);
          if (pre_reclaimation_size <= reservation_size) {
            tracked_bytes = 0;                              // was fully inside reservation
          } else if (post_reclaimation_size < reservation_size) {
            tracked_bytes = static_cast<std::size_t>(  // only the overflow portion
              pre_reclaimation_size - reservation_size);
          }
          // else: was fully beyond reservation -> credit full (tracked_bytes).
        }
        // else: reservation already released. release_reservation credited
        // only the unconsumed portion; this allocation's bytes are still owed
        // to the counter, so credit the FULL amount (tracked_bytes unchanged).
      }
    }
    // Free outside the lock: hipFreeHost may block, and the map entry is
    // already removed so no other thread can touch this allocation.
    if (ptr) {
      hipFreeHost(ptr);
      do_release(tracked_bytes);
    }
    alloc.reset();
  }

  void* allocate(rmm::cuda_stream_view, std::size_t bytes,
                 std::size_t /*alignment*/ = 0) {
    if (bytes == 0) return nullptr;
    // Unmanaged single-block allocation: full charge to the counter (no
    // reservation association). Bounded by mem_limit_ the same way as
    // allocate_multiple_blocks.
    if (!do_reserve(bytes, mem_limit_)) {
      throw std::runtime_error(
        std::string("cuCascade: fixed_size_host_memory_resource OOM: cannot allocate ") +
        std::to_string(bytes) + " bytes within memory limit " +
        std::to_string(mem_limit_));
    }
    void* ptr = nullptr;
    hipError_t err = hipMallocHost(&ptr, bytes);
    if (err != hipSuccess || ptr == nullptr) {
      do_release(bytes);
      throw std::runtime_error(
        std::string("cuCascade: hipMallocHost failed in "
                    "fixed_size_host_memory_resource::allocate: ") +
        hipGetErrorString(err));
    }
    return ptr;
  }
  void deallocate(rmm::cuda_stream_view, void* ptr, std::size_t bytes,
                  std::size_t /*alignment*/ = 0) {
    if (ptr) {
      hipFreeHost(ptr);
      do_release(bytes);
    }
  }

  operator rmm::device_async_resource_ref() const {
    return upstream_;
  }

 private:
  /// Per-reservation usage tracker. Tracks how many bytes have actually been
  /// allocated under this reservation, so the overflow charge (and its
  /// symmetric credit) can be computed against the reservation size.
  struct allocation_tracker {
    std::atomic<int64_t> allocated_bytes{0};
  };

  /// Round `bytes` up to the next multiple of `block_size_`.
  static std::size_t align_up(std::size_t bytes, std::size_t block_size) {
    return ((bytes + block_size - 1) / block_size) * block_size;
  }

  /// Single-counter atomic check-and-charge. Atomically adds `bytes` to
  /// `total_allocated_` only if `current + bytes <= mem_limit` (when
  /// mem_limit != 0). Returns false if the charge would exceed the limit.
  /// Overflow-safe: `bytes > mem_limit` is rejected before the subtraction.
  /// No mutex required — the CAS is the synchronization.
  bool do_reserve(std::size_t bytes, std::size_t mem_limit) {
    if (bytes == 0) return true;
    if (mem_limit != 0) {
      if (bytes > mem_limit) return false;
      std::size_t cur = total_allocated_.load(std::memory_order_relaxed);
      while (true) {
        if (cur > mem_limit - bytes) return false;  // would exceed limit
        if (total_allocated_.compare_exchange_weak(cur, cur + bytes,
                std::memory_order_relaxed)) {
          break;
        }
        // cur reloaded by the failed CAS; retry.
      }
    } else {
      // Unlimited mode (mem_limit_ == 0): no admission cap, just charge.
      total_allocated_.fetch_add(bytes, std::memory_order_relaxed);
    }
    update_peak();
    return true;
  }

  /// Credit `bytes` back to the single counter, clamped at 0 so a
  /// bookkeeping asymmetry can never underflow the counter.
  void do_release(std::size_t bytes) {
    if (bytes == 0) return;
    std::size_t cur = total_allocated_.load(std::memory_order_relaxed);
    while (true) {
      std::size_t next = (cur >= bytes) ? (cur - bytes) : 0;
      if (total_allocated_.compare_exchange_weak(cur, next,
              std::memory_order_relaxed)) {
        break;
      }
    }
  }

  void update_peak() {
    std::size_t now = total_allocated_.load(std::memory_order_relaxed);
    std::size_t peak = peak_allocated_.load(std::memory_order_relaxed);
    while (now > peak && !peak_allocated_.compare_exchange_weak(peak, now,
             std::memory_order_relaxed)) {}
  }

  /// Release a reservation: credit only the UNCONSUMED portion
  /// (`arena_size - allocated_under_it`) back to the single counter, then
  /// drop the tracker. Fully-consumed reservations credit nothing (their
  /// bytes were already returned by the dealloc path).
  void release_reservation(reserved_arena* arena) {
    if (!arena) return;
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = active_reservations_.find(arena);
    if (it == active_reservations_.end()) return;  // already released
    auto current = static_cast<int64_t>(
      std::max<int64_t>(0, it->second.allocated_bytes.load(std::memory_order_relaxed)));
    auto arena_size = static_cast<int64_t>(arena->size());
    std::size_t reclaimed_bytes =
      arena_size > current ? static_cast<std::size_t>(arena_size - current) : 0;
    active_reservations_.erase(it);
    // Credit outside the map lookup, but we still hold the mutex (lk).
    do_release(reclaimed_bytes);
  }

  int32_t device_id_{0};
  rmm::device_async_resource_ref upstream_;
  std::size_t mem_limit_{0};
  std::size_t block_size_{default_block_size};

  /// THE single committed-bytes counter. Includes both reserved (pre-charged)
  /// and allocated bytes; an allocation under a reservation only adds the
  /// overflow beyond the reservation, so a byte is counted exactly once.
  std::atomic<std::size_t> total_allocated_{0};
  std::atomic<std::size_t> peak_allocated_{0};

  /// Single mutex guarding both the reservation tracker map and the live
  /// allocation map (and, by extension, the tracker charge/credit spans so a
  /// concurrent release_reservation cannot destroy a tracker mid-charge).
  /// Matches the real cuCascade's single _mutex.
  mutable std::mutex mutex_;
  /// Per-reservation usage trackers, keyed by the arena pointer. The arena
  /// pointer is stable for the reservation's lifetime (owned by the
  /// reservation handle), so keying by it is safe.
  std::unordered_map<reserved_arena*, allocation_tracker> active_reservations_;
  /// Tracks live multiple_blocks_allocations so their backing hipMallocHost
  /// pointer can be freed in deallocate_multiple_blocks (the allocation
  /// object owns N logical blocks but one physical allocation).
  std::unordered_map<uintptr_t, std::pair<void*, std::size_t>> live_allocs_;

  /// reserved_arena impl that reports a byte count.
  struct reserved_arena_bytes : reserved_arena {
    explicit reserved_arena_bytes(std::size_t n) : bytes_(n) {}
    std::size_t size() const override { return bytes_; }
    void grow_by(std::size_t n) override { bytes_ += n; }
    std::size_t bytes_{0};
  };
};

}  // namespace cucascade::memory
