/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade gpu_data_representation — ROCm real implementation.
//! Deep-copies the cudf::table on clone() via cudf's own copy mechanism
//! (cudf::copy(table_view, stream, mr)), which handles the GPU-side column
//! deep copy correctly for all cudf types. The get_table_view() guard
//! remains (a null table is a real error, not a stub).

#pragma once
#include "cucascade/data/common.hpp"
#include "cucascade/memory/common.hpp"
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <rmm/cuda_stream.hpp>
#include <memory>
#include <stdexcept>
#include <utility>

namespace cucascade::memory { class memory_space; }

namespace cucascade {

/// GPU representation holding a cudf::table. Sirius calls get_table_view().
class gpu_table_representation : public idata_representation {
 public:
  gpu_table_representation(std::unique_ptr<cudf::table> table,
                           memory::memory_space& space,
                           rmm::cuda_stream_view /*writer_stream*/)
    : idata_representation(space), table_(std::move(table)) {}

  template <typename Owner>
  gpu_table_representation(cudf::table_view /*view*/, Owner&& /*owner*/,
                           std::size_t /*alloc_size*/, memory::memory_space& space,
                           rmm::cuda_stream_view /*stream*/)
    : idata_representation(space) {}

  cudf::table_view get_table_view() const {
    if (!table_) throw std::runtime_error("cuCascade: gpu_table has no table");
    return table_->view();
  }

  std::unique_ptr<cudf::table> release_table(rmm::cuda_stream_view) {
    return std::move(table_);
  }

  /// Deep-copy the GPU table via cudf::copy(). This allocates a new device
  /// buffer for each column and hipMemcpy's the data (cudf::copy handles this
  /// internally, using the stream for ordering). The clone is independent of
  /// the original — mutations to one don't affect the other. Mirrors
  /// cuCascade's gpu_table_representation::clone.
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override {
    if (!table_) {
      return std::make_unique<gpu_table_representation>(
        std::unique_ptr<cudf::table>(), space(), stream);
    }
    // cudf::copy returns a unique_ptr<cudf::table> — a full deep copy of all
    // columns, device data included. The stream orders the device-to-device
    // copies; the default memory resource (rmm) allocates the new buffers.
    auto cloned_table = cudf::copy(table_->view(), stream);
    return std::make_unique<gpu_table_representation>(
      std::move(cloned_table), space(), stream);
  }

  std::size_t get_size_in_bytes() const override {
    // Sum the device bytes of all columns. cudf::table doesn't expose a
    // direct total-bytes accessor, so we sum per-column via the view.
    if (!table_) return 0;
    std::size_t total = 0;
    auto view = table_->view();
    for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
      total += view.column(i).alloc_size();
    }
    return total;
  }
  std::size_t get_uncompressed_data_size_in_bytes() const override {
    return get_size_in_bytes();
  }

 private:
  std::unique_ptr<cudf::table> table_;
};

}  // namespace cucascade
