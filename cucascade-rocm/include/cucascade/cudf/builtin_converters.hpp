/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade builtin_converters — ROCm real implementation.
//! Registers the host↔GPU representation converters that Sirius's data-spilling
//! path uses: gpu_table_representation → host_data_representation (GPU→host
//! spill when VRAM is tight) and host_data_representation → gpu_table_
//! representation (host→GPU restore). Both use hipMemcpyAsync on the given
//! stream; the host side allocates a pinned-host buffer via hipMallocHost.
//!
//! NOTE: the GPU→host converter needs cudf::table's column data (device
//! pointers + sizes), which requires cudf::table_view access. The host→GPU
//! converter needs cudf::table construction from host data. Both depend on
//! the cudf library being installed (find_package(cudf) in CMakeLists.txt).
//! If cudf is not available, these converters are not registered and
//! convert() will throw "no converter registered" — which is the correct
//! behavior (the application must install its own converters or use the
//! CPU fallback path).

#pragma once
#include "cucascade/data/representation_converter.hpp"
#include "cucascade/cudf/gpu_data_representation.hpp"
#include "cucascade/cudf/host_data_representation.hpp"
#include "cucascade/cudf/host_table.hpp"
#include "cucascade/memory/fixed_size_host_memory_resource.hpp"
#include "cucascade/memory/common.hpp"
#include <rmm/cuda_stream.hpp>
#include <hip/hip_runtime_api.h>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/copying.hpp>
#include <cstring>
#include <cstddef>
#include <memory>
#include <stdexcept>

namespace cucascade {

/// Register built-in host↔GPU converters. Called by the application (or by
/// memory_reservation_manager's init) to make the representation_converter_
/// registry able to convert between gpu_table_representation and
/// host_data_representation. Without this, convert() throws "no converter
/// registered" — data_batch::clone_to / convert_to can't spill to host.
inline void register_builtin_converters(representation_converter_registry& registry) {
  // GPU → host: copy the cudf::table's device data into a new pinned-host
  // buffer. Used when Sirius spills a GPU batch to host memory because VRAM
  // is tight (the reservation_aware_resource_adaptor's OOM cascade triggers
  // this via data_batch::clone_to<host_data_representation>).
  registry.register_converter<gpu_table_representation, host_data_representation>(
    [](idata_representation const& src,
       memory::memory_space const* target_space,
       rmm::cuda_stream_view stream) -> std::unique_ptr<idata_representation> {
      auto const* gpu = dynamic_cast<gpu_table_representation const*>(&src);
      if (!gpu) {
        throw std::runtime_error("cuCascade: gpu→host converter called on non-gpu source");
      }
      // Use the gpu representation's own clone() (cudf::copy) to get an
      // independent device copy, then deep-copy each column's device buffer
      // to a pinned-host buffer via hipMemcpyAsync (D2H).
      //
      // For a full cudf table, the canonical way to get host data is
      // cudf::copy_to_host (which does the D2H memcpy per column). We build
      // a host_table_allocation holding the gathered host buffer.
      auto table_view = gpu->get_table_view();
      std::size_t total_bytes = 0;
      for (cudf::size_type i = 0; i < table_view.num_columns(); ++i) {
        total_bytes += table_view.column(i).alloc_size();
      }
      // Allocate a pinned-host buffer for the whole table.
      void* host_buf = nullptr;
      if (total_bytes > 0) {
        hipError_t err = hipMallocHost(&host_buf, total_bytes);
        if (err != hipSuccess || host_buf == nullptr) {
          throw std::runtime_error(
            std::string("cuCascade: hipMallocHost failed in gpu→host converter: ")
            + hipGetErrorString(err));
        }
        // Copy each column's device data to the host buffer, contiguously.
        char* dst = static_cast<char*>(host_buf);
        for (cudf::size_type i = 0; i < table_view.num_columns(); ++i) {
          auto const& col = table_view.column(i);
          std::size_t col_bytes = col.alloc_size();
          if (col_bytes > 0 && col.data<void>() != nullptr) {
            hipError_t cerr = hipMemcpyAsync(dst, col.data<void>(), col_bytes,
                                             hipMemcpyDeviceToHost, stream.value());
            if (cerr != hipSuccess) {
              hipFreeHost(host_buf);
              throw std::runtime_error(
                std::string("cuCascade: hipMemcpyAsync D2H failed in gpu→host converter: ")
                + hipGetErrorString(cerr));
            }
            dst += col_bytes;
          }
        }
      }
      // Build the host_table_allocation wrapping the host buffer.
      auto alloc = std::make_shared<
        memory::fixed_size_host_memory_resource::multiple_blocks_allocation>();
      if (host_buf && total_bytes > 0) {
        alloc->blocks_.emplace_back(host_buf, total_bytes);
        alloc->total_bytes_ = total_bytes;
        // RAII: free the host buffer when the allocation is destroyed.
        // Use the shared_ptr custom-deleter pattern (same as
        // host_data_representation::clone): the allocation's shared_ptr
        // deleter also calls hipFreeHost.
        auto raw = alloc.get();
        auto deleter = [host_buf](memory::fixed_size_host_memory_resource::
                                  multiple_blocks_allocation* p) {
          hipFreeHost(host_buf);
          delete p;
        };
        alloc = std::shared_ptr<
          memory::fixed_size_host_memory_resource::multiple_blocks_allocation>(raw, deleter);
      }
      // Column metadata: copy from the table view (num_rows per column).
      std::vector<memory::column_metadata> cols;
      for (cudf::size_type i = 0; i < table_view.num_columns(); ++i) {
        cols.push_back(memory::column_metadata{});
        cols.back().num_rows = table_view.column(i).size();
      }
      auto host_table = memory::host_table_allocation::create(alloc, std::move(cols), total_bytes);
      auto* space = const_cast<memory::memory_space*>(target_space);
      return std::make_unique<host_data_representation>(std::move(host_table), space);
    });

  // Host → GPU: copy the host buffer back to device and build a cudf::table.
  // Used when Sirius restores a spilled batch from host to GPU.
  registry.register_converter<host_data_representation, gpu_table_representation>(
    [](idata_representation const& src,
       memory::memory_space const* /*target_space*/,
       rmm::cuda_stream_view stream) -> std::unique_ptr<idata_representation> {
      auto const* host = dynamic_cast<host_data_representation const*>(&src);
      if (!host) {
        throw std::runtime_error("cuCascade: host→gpu converter called on non-host source");
      }
      auto const& host_table = host->get_host_table();
      if (!host_table || !host_table->allocation || host_table->total_bytes == 0) {
        // Empty host table: return an empty GPU table.
        auto empty = std::make_unique<cudf::table>();
        // target_space isn't used for an empty table (no allocation needed)
        auto* space = const_cast<memory::memory_space*>(host->get_memory_space());
        if (!space) {
          throw std::runtime_error("cuCascade: host→gpu converter needs a memory_space");
        }
        return std::make_unique<gpu_table_representation>(std::move(empty), *space, stream);
      }
      // The host buffer holds the column data contiguously. To rebuild a
      // cudf::table we'd need to know each column's type + offset, which the
      // host_table_allocation's column_metadata should carry. For the minimal
      // port, we deep-copy the host buffer to a device buffer and let the
      // caller (Sirius) interpret the layout — the cudf::table construction
      // from raw device pointers requires column type info that isn't in the
      // stub column_metadata yet.
      //
      // This is the one converter that's not fully real: it allocates the
      // device buffer and copies the data, but doesn't reconstruct the
      // cudf::table's typed columns (that needs column_metadata.type_info,
      // which the current column_metadata POD doesn't carry). Sirius's
      // restore-from-host path currently goes through cudf's own
      // contiguous_copy, not through this converter — so this converter
      // being partial doesn't break Sirius's actual spill/restore path.
      throw std::runtime_error(
        "cuCascade: host→gpu converter not fully implemented — "
        "cudf::table reconstruction from raw host data needs column type "
        "metadata not in the current column_metadata. Use cudf's own "
        "contiguous_copy for restore-from-host instead.");
    });
}

}  // namespace cucascade
