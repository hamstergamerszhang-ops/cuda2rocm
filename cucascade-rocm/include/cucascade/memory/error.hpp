/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! cuCascade memory error types — ROCm stub.

#pragma once

// Same stub-gate as common.hpp -- skip the real header when a caller has
// already stubbed rmm::out_of_memory (see test_cucascade_compile.cpp).
#ifndef RMM_DEVICE_ASYNC_RESOURCE_REF_STUBBED
#include <rmm/detail/error.hpp>  // rmm::out_of_memory, rmm::bad_alloc (hipMM has no rmm/out_of_memory.hpp)
#endif
#include <cuda_runtime.h>  // cudaMemPool_t (shim → hipMemPool_t)
#include <cstddef>
#include <string>
#include <string_view>

namespace cucascade::memory {

/// Memory operation error category.
enum class MemoryError {
  SUCCESS,
  ALLOCATION_FAILED,
  LIMIT_EXCEEDED,
  POOL_EXHAUSTED,
  SIZE
};

/// Exception thrown when cuCascade's memory reservation fails.
/// Derives from rmm::out_of_memory so Sirius's catch(rmm::out_of_memory&)
/// + dynamic_cast<cucascade_out_of_memory> works.
struct cucascade_out_of_memory : public rmm::out_of_memory {
  cucascade_out_of_memory(std::string_view message, MemoryError error_kind,
                          std::size_t requested_bytes, std::size_t global_usage,
                          cudaMemPool_t pool_handle)
    : rmm::out_of_memory(std::string(message)),
      error_kind(error_kind),
      requested_bytes(requested_bytes),
      global_usage(global_usage),
      pool_handle(pool_handle) {}

  const MemoryError error_kind;
  const std::size_t requested_bytes;
  const std::size_t global_usage;
  const cudaMemPool_t pool_handle;
};

}  // namespace cucascade::memory
