/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade fixed_size_host_memory_resource — ROCm real implementation.
//! Pinned host RAM pool backed by hipMallocHost / hipFreeHost.
//! multiple_blocks_allocation is NESTED inside the class (Sirius references
//! it as fixed_size_host_memory_resource::multiple_blocks_allocation).

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
    return total_allocated_.load(std::memory_order_relaxed) / block_size_;
  }
  std::size_t get_total_allocated_bytes() const {
    return total_allocated_.load(std::memory_order_relaxed);
  }
  std::size_t get_total_reserved_bytes() const {
    return reserved_bytes_.load(std::memory_order_relaxed);
  }
  std::size_t get_active_reservation_count() const {
    return active_reservations_.load(std::memory_order_relaxed);
  }
  std::size_t get_peak_total_allocated_bytes() const {
    return peak_allocated_.load(std::memory_order_relaxed);
  }

  /// Reserve `bytes` of host capacity. Returns a reservation handle whose
  /// release callback decrements the reserved/active counters.
  std::unique_ptr<reservation> reserve(std::size_t bytes,
                                       notification_channel* /*notifier*/ = nullptr) {
    if (mem_limit_ != 0) {
      std::size_t used = total_allocated_.load(std::memory_order_relaxed);
      if (used + bytes > mem_limit_) {
        throw std::runtime_error(
          "cuCascade: fixed_size_host_memory_resource::reserve exceeds mem_limit");
      }
    }
    reserved_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    active_reservations_.fetch_add(1, std::memory_order_relaxed);
    // The reservation holds a pointer back to this resource so its release
    // callback can decrement the counters. memory_space owns this resource
    // for the engine's lifetime, so the raw pointer is safe.
    auto* self = this;
    auto on_release = [self, bytes]() {
      self->reserved_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
      self->active_reservations_.fetch_sub(1, std::memory_order_relaxed);
    };
    auto arena = std::make_unique<reserved_arena_bytes>(bytes);
    // Host-only reservations are not backed by a memory_space; create the
    // reservation with a null space so tier()/device_id() return HOST/0 and
    // get_memory_space() throws if called.
    return reservation::create(std::move(arena), std::move(on_release));
  }

  /// Allocate `bytes` of pinned host memory, rounded up to block_size_.
  fixed_multiple_blocks_allocation allocate_multiple_blocks(std::size_t bytes,
                                                            reservation* /*res*/ = nullptr) {
    if (bytes == 0) {
      return std::make_unique<multiple_blocks_allocation>();
    }
    std::size_t n_blocks = (bytes + block_size_ - 1) / block_size_;
    std::size_t alloc_bytes = n_blocks * block_size_;
    void* ptr = nullptr;
    hipError_t err = hipMallocHost(&ptr, alloc_bytes);
    if (err != hipSuccess || ptr == nullptr) {
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
    total_allocated_.fetch_add(alloc_bytes, std::memory_order_relaxed);
    std::size_t now = total_allocated_.load(std::memory_order_relaxed);
    std::size_t peak = peak_allocated_.load(std::memory_order_relaxed);
    while (now > peak && !peak_allocated_.compare_exchange_weak(peak, now,
             std::memory_order_relaxed)) {}
    // Track the raw pointer + size for deallocate_multiple_blocks.
    {
      std::lock_guard<std::mutex> lk(live_allocs_mutex_);
      live_allocs_[reinterpret_cast<uintptr_t>(alloc.get())] = {ptr, alloc_bytes};
    }
    return alloc;
  }

  void deallocate_multiple_blocks(fixed_multiple_blocks_allocation alloc) {
    void* ptr = nullptr;
    std::size_t bytes = 0;
    {
      std::lock_guard<std::mutex> lk(live_allocs_mutex_);
      auto it = live_allocs_.find(reinterpret_cast<uintptr_t>(alloc.get()));
      if (it != live_allocs_.end()) {
        ptr = it->second.first;
        bytes = it->second.second;
        live_allocs_.erase(it);
      }
    }
    // Free outside the lock: hipFreeHost may block, and the map entry is
    // already removed so no other thread can touch this allocation.
    if (ptr) {
      hipFreeHost(ptr);
      total_allocated_.fetch_sub(bytes, std::memory_order_relaxed);
    }
    alloc.reset();
  }

  void* allocate(rmm::cuda_stream_view, std::size_t bytes,
                 std::size_t /*alignment*/ = 0) {
    if (bytes == 0) return nullptr;
    void* ptr = nullptr;
    hipError_t err = hipMallocHost(&ptr, bytes);
    if (err != hipSuccess || ptr == nullptr) {
      throw std::runtime_error(
        std::string("cuCascade: hipMallocHost failed in "
                    "fixed_size_host_memory_resource::allocate: ") +
        hipGetErrorString(err));
    }
    total_allocated_.fetch_add(bytes, std::memory_order_relaxed);
    std::size_t now = total_allocated_.load(std::memory_order_relaxed);
    std::size_t peak = peak_allocated_.load(std::memory_order_relaxed);
    while (now > peak && !peak_allocated_.compare_exchange_weak(peak, now,
             std::memory_order_relaxed)) {}
    return ptr;
  }
  void deallocate(rmm::cuda_stream_view, void* ptr, std::size_t bytes,
                  std::size_t /*alignment*/ = 0) {
    if (ptr) {
      hipFreeHost(ptr);
      total_allocated_.fetch_sub(bytes, std::memory_order_relaxed);
    }
  }

  operator rmm::device_async_resource_ref() const {
    return upstream_;
  }

 private:
  int32_t device_id_{0};
  rmm::device_async_resource_ref upstream_;
  std::size_t mem_limit_{0};
  std::size_t block_size_{default_block_size};
  std::atomic<std::size_t> total_allocated_{0};
  std::atomic<std::size_t> reserved_bytes_{0};
  std::atomic<std::size_t> active_reservations_{0};
  std::atomic<std::size_t> peak_allocated_{0};

  /// Guards live_allocs_ (written in allocate_multiple_blocks, read/erased in
  /// deallocate_multiple_blocks — previously unlocked, a data race).
  std::mutex live_allocs_mutex_;
  /// Tracks live multiple_blocks_allocations so their backing hipMallocHost
  /// pointer can be freed in deallocate_multiple_blocks (the allocation object
  /// owns N logical blocks but one physical allocation).
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
