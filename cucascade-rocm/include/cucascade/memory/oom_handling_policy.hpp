/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file cuCascade OOM handling policy — ROCm stub.
//! Real virtual base — Sirius's defragmenter_oom_policy inherits this.

#pragma once

#include "cucascade/memory/error.hpp"
// Same stub-gate as common.hpp -- skip the real header when a caller has
// already stubbed rmm::cuda_stream_view (see test_cucascade_compile.cpp).
#ifndef RMM_DEVICE_ASYNC_RESOURCE_REF_STUBBED
#include <rmm/cuda_stream.hpp>
#endif
#include <cstddef>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>

namespace cucascade::memory {

class oom_handling_policy {
 public:
  using RetryFunc = std::function<void*(std::size_t, rmm::cuda_stream_view)>;

  virtual ~oom_handling_policy() = default;

  void* handle_oom(std::size_t requested_bytes, rmm::cuda_stream_view stream,
                   std::exception_ptr eptr, RetryFunc retry) {
    return do_handle_oom(requested_bytes, stream, eptr, retry);
  }

  virtual std::string get_policy_name() const noexcept = 0;

 protected:
  virtual void* do_handle_oom(std::size_t, rmm::cuda_stream_view,
                              std::exception_ptr, RetryFunc) = 0;
};

class throw_on_oom_policy : public oom_handling_policy {
 public:
  std::string get_policy_name() const noexcept override { return "throw_on_oom"; }
 protected:
  void* do_handle_oom(std::size_t, rmm::cuda_stream_view,
                      std::exception_ptr eptr, RetryFunc) override {
    std::rethrow_exception(eptr);
  }
};

inline std::unique_ptr<oom_handling_policy> make_default_oom_policy() {
  return std::make_unique<throw_on_oom_policy>();
}

}  // namespace cucascade::memory
