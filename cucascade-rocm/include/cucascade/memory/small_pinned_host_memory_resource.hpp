/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade small_pinned_host_memory_resource — ROCm real implementation.
//! Thin wrapper around fixed_size_host_memory_resource for small (< 1 MiB)
//! pinned-host allocations. The real fixed_size pool handles hipMallocHost/
//! hipFreeHost, reservation tracking, and block-granular allocation; this
//! class just forwards to it with a smaller block size suited for small allocs.
//! Mirrors cuCascade's small_pinned_host_memory_resource, which delegates to
//! the same underlying pinned-host pool with a smaller slab.

#pragma once
#include "cucascade/memory/fixed_size_host_memory_resource.hpp"
#include <rmm/cuda_stream.hpp>
#include <rmm/resource_ref.hpp>
#include <cstddef>
#include <stdexcept>

namespace cucascade::memory {

class small_pinned_host_memory_resource {
 public:
  static constexpr std::size_t MAX_SLAB_SIZE = 1 << 20; // 1 MiB

  small_pinned_host_memory_resource(int32_t device_id,
                                    rmm::device_async_resource_ref upstream,
                                    std::size_t mem_limit = 0,
                                    std::size_t slab_size = MAX_SLAB_SIZE)
    // Delegate to the real fixed_size_host_memory_resource with a small block
    // size. The fixed_size pool rounds allocations up to block_size_ and tracks
    // them against the reservation counter — identical semantics to the real
    // cuCascade, just with a smaller granularity for small allocs.
    : pool_(device_id, upstream, mem_limit, 0,
            slab_size, fixed_size_host_memory_resource::default_pool_size,
            fixed_size_host_memory_resource::default_initial_number_pools) {}

  void* allocate(rmm::cuda_stream_view stream, std::size_t bytes,
                 std::size_t alignment = 0) {
    return pool_.allocate(stream, bytes, alignment);
  }
  void deallocate(rmm::cuda_stream_view stream, void* ptr, std::size_t bytes,
                  std::size_t alignment = 0) {
    pool_.deallocate(stream, ptr, bytes, alignment);
  }

  operator rmm::device_async_resource_ref() const {
    return static_cast<rmm::device_async_resource_ref>(pool_);
  }

 private:
  fixed_size_host_memory_resource pool_;
};

}  // namespace cucascade::memory
