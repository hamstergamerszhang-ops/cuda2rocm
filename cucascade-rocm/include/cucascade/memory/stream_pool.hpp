/*
 * Copyright 2026, rocm-cuda-compat contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file cuCascade stream pool — ROCm real implementation.
//! Uses hipStreamCreateWithFlags for real stream management.

#pragma once

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_device.hpp>
#include <cuda_runtime.h>  // shim → hip runtime
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace cucascade::memory {

class exclusive_stream_pool;

/// A borrowed stream RAII wrapper. Holds a non-owning view of a stream owned
/// by the pool that lent it, plus the pool index so the stream is returned on
/// destruction.
class borrowed_stream {
 public:
  borrowed_stream() = default;
  borrowed_stream(rmm::cuda_stream_view view, std::size_t index, exclusive_stream_pool* pool)
    : view_(view), index_(index), pool_(pool) {}

  borrowed_stream(borrowed_stream const&) = delete;
  borrowed_stream& operator=(borrowed_stream const&) = delete;
  borrowed_stream(borrowed_stream&&) noexcept = default;
  borrowed_stream& operator=(borrowed_stream&&) noexcept = default;

  ~borrowed_stream() { release(); }

  rmm::cuda_stream_view get() const { return view_; }
  operator rmm::cuda_stream_view() const { return view_; }

  /// Relinquish the borrowed stream back to its pool early. After release()
  /// this wrapper no longer owns a stream from the pool.
  void release() noexcept;

 private:
  rmm::cuda_stream_view view_{};
  std::size_t index_{static_cast<std::size_t>(-1)};
  exclusive_stream_pool* pool_{nullptr};
};

/// Exclusive stream pool — manages a set of HIP streams.
/// GROW policy: creates streams on demand. BLOCK: reuses existing.
class exclusive_stream_pool {
 public:
  enum class stream_acquire_policy { GROW, BLOCK };

  explicit exclusive_stream_pool(rmm::cuda_device_id device_id, std::size_t pool_size = 0,
                                 rmm::cuda_stream::flags flags = {})
    : device_id_(device_id), pool_size_(pool_size), flags_(flags) {
    rmm::cuda_set_device_raii set_device{device_id_};
    // Pre-create pool_size streams
    for (std::size_t i = 0; i < pool_size; ++i) {
      hipStream_t s = nullptr;
      if (hipStreamCreateWithFlags(&s, flags_.value()) == hipSuccess) {
        streams_.push_back(s);
        free_.push_back(true);
      }
    }
  }

  ~exclusive_stream_pool() {
    rmm::cuda_set_device_raii set_device{device_id_};
    for (auto s : streams_) {
      if (s) hipStreamDestroy(s);
    }
  }

  borrowed_stream acquire_stream(stream_acquire_policy policy = stream_acquire_policy::GROW) {
    // Try to find a free stream
    for (std::size_t i = 0; i < free_.size(); ++i) {
      if (free_[i]) {
        free_[i] = false;
        return borrowed_stream{rmm::cuda_stream_view{streams_[i]}, i, this};
      }
    }
    // GROW: create a new stream on this pool's device
    if (policy == stream_acquire_policy::GROW) {
      rmm::cuda_set_device_raii set_device{device_id_};
      hipStream_t s = nullptr;
      if (hipStreamCreateWithFlags(&s, flags_.value()) == hipSuccess) {
        std::size_t index = streams_.size();
        streams_.push_back(s);
        free_.push_back(false);
        return borrowed_stream{rmm::cuda_stream_view{s}, index, this};
      }
    }
    // Fallback: per-thread default stream
    return borrowed_stream{rmm::cuda_stream_per_thread, static_cast<std::size_t>(-1), nullptr};
  }

  void return_stream(std::size_t index) {
    if (index < free_.size()) { free_[index] = true; }
  }

  std::size_t size() const { return streams_.size(); }
  rmm::cuda_device_id get_device_id() const { return device_id_; }

 private:
  rmm::cuda_device_id device_id_;
  std::size_t pool_size_{0};
  rmm::cuda_stream::flags flags_{};
  std::vector<hipStream_t> streams_;
  std::vector<bool> free_;
};

inline void borrowed_stream::release() noexcept {
  if (pool_ && index_ != static_cast<std::size_t>(-1)) {
    pool_->return_stream(index_);
    pool_ = nullptr;
    index_ = static_cast<std::size_t>(-1);
    view_ = rmm::cuda_stream_view{};
  }
}

}  // namespace cucascade::memory
