/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade host_data_representation — ROCm real implementation.
//! Deep-copies the host_table_allocation (pinned host buffer + column metadata)
//! on clone(). The buffer copy uses hipMemcpyAsync (host-to-host on the given
//! stream); the column metadata vector is copied by value (it's POD).

#pragma once
#include "cucascade/data/common.hpp"
#include "cucascade/cudf/host_table.hpp"
#include "cucascade/memory/common.hpp"
#include "cucascade/memory/fixed_size_host_memory_resource.hpp"
#include <rmm/cuda_stream.hpp>
#include <hip/hip_runtime_api.h>
#include <cstring>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

namespace cucascade::memory { class memory_space; }

namespace cucascade {

class host_data_representation : public idata_representation {
 public:
  host_data_representation(std::unique_ptr<memory::host_table_allocation> table,
                           memory::memory_space* space)
    : idata_representation(*space), table_(std::move(table)) {}

  std::unique_ptr<memory::host_table_allocation> const& get_host_table() const { return table_; }

  std::size_t num_columns() const { return table_ ? table_->num_columns() : 0; }
  std::size_t column_size(std::size_t i) const { return table_ ? table_->column_size(i) : 0; }
  void slice(std::span<std::size_t> offsets) { if (table_) table_->slice(offsets); }

  /// Deep-copy the host table: allocate a new pinned-host buffer of the same
  /// total size, hipMemcpyAsync the data (host-to-host on the given stream),
  /// and copy the column metadata vector by value. Returns a new
  /// host_data_representation wrapping the copied allocation. The clone owns
  /// its own buffer independent of the original (a mutation to one does not
  /// affect the other). Mirrors cuCascade's host_data_representation::clone.
  ///
  /// Ownership: the raw hipMallocHost buffer is freed when the
  /// multiple_blocks_allocation that holds it is destroyed. We achieve this
  /// by constructing the allocation on the heap (new) and wrapping it in a
  /// shared_ptr with a custom deleter that calls hipFreeHost(new_buf) AND
  /// deletes the allocation. This is the same RAII pattern used in
  /// builtin_converters.hpp's gpu→host converter. The blocks_ vector inside
  /// the allocation holds a block{new_buf, total_bytes} so Sirius's
  /// get_blocks() sees the buffer; the deleter ensures it's freed.
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override {
    if (!table_) {
      return std::make_unique<host_data_representation>(
        std::unique_ptr<memory::host_table_allocation>(), &space());
    }

    auto cols = table_->columns;
    std::size_t total_bytes = table_->total_bytes;

    void* new_buf = nullptr;
    if (total_bytes > 0) {
      hipError_t err = hipMallocHost(&new_buf, total_bytes);
      if (err != hipSuccess || new_buf == nullptr) {
        throw std::runtime_error(
          std::string("cuCascade: hipMallocHost failed in host_data_representation::clone: ")
          + hipGetErrorString(err));
      }
      char* dst = static_cast<char*>(new_buf);
      if (table_->allocation) {
        for (auto const& blk : table_->allocation->get_blocks()) {
          if (blk.ptr && blk.bytes > 0) {
            hipError_t cerr = hipMemcpyAsync(dst, blk.ptr, blk.bytes,
                                             hipMemcpyHostToHost, stream.value());
            if (cerr != hipSuccess) {
              hipFreeHost(new_buf);
              throw std::runtime_error(
                std::string("cuCascade: hipMemcpyAsync failed in host_data_representation::clone: ")
                + hipGetErrorString(cerr));
            }
            dst += blk.bytes;
          }
        }
      } else {
        std::memset(new_buf, 0, total_bytes);
      }
    }

    // Build the cloned allocation. Construct on the heap, then wrap in a
    // shared_ptr with a custom deleter that frees the hipMallocHost buffer
    // when the allocation's refcount drops to zero. This is the only correct
    // way to tie a raw HIP buffer's lifetime to a multiple_blocks_allocation
    // without modifying fixed_size_host_memory_resource.hpp (whose blocks_
    // are plain POD structs with no deleter).
    using Alloc = memory::fixed_size_host_memory_resource::multiple_blocks_allocation;
    Alloc* raw_alloc = new Alloc();
    std::shared_ptr<Alloc> new_alloc(raw_alloc, [new_buf](Alloc* p) {
      if (new_buf) hipFreeHost(new_buf);
      delete p;
    });
    if (total_bytes > 0 && new_buf) {
      raw_alloc->blocks_.emplace_back(new_buf, total_bytes);
      raw_alloc->total_bytes_ = total_bytes;
    }

    auto cloned_table = memory::host_table_allocation::create(new_alloc, std::move(cols), total_bytes);
    return std::make_unique<host_data_representation>(std::move(cloned_table), &space());
  }

  std::size_t get_size_in_bytes() const override { return table_ ? table_->total_bytes : 0; }
  std::size_t get_uncompressed_data_size_in_bytes() const override { return get_size_in_bytes(); }

 private:
  std::unique_ptr<memory::host_table_allocation> table_;
};

class host_data_packed_representation : public idata_representation {
 public:
  host_data_packed_representation(std::unique_ptr<memory::host_table_packed_allocation> alloc,
                                  memory::memory_space& space)
    : idata_representation(space), alloc_(std::move(alloc)) {}

  /// Deep-copy the packed allocation. The packed form is just a total_bytes
  /// counter (no separate column metadata), so the clone is a new
  /// host_table_packed_allocation with the same byte count.
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view) override {
    auto cloned = std::make_unique<memory::host_table_packed_allocation>();
    if (alloc_) cloned->total_bytes = alloc_->total_bytes;
    return std::make_unique<host_data_packed_representation>(std::move(cloned), space());
  }
  std::size_t get_size_in_bytes() const override { return alloc_ ? alloc_->total_bytes : 0; }
  std::size_t get_uncompressed_data_size_in_bytes() const override { return get_size_in_bytes(); }

 private:
  std::unique_ptr<memory::host_table_packed_allocation> alloc_;
};

}  // namespace cucascade
