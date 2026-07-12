/*
 * Copyright 2026, rocm-cuda-compat contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! cuco::extent — size wrapper for bloom_filter and static_set.
//! Matches NVIDIA cuCollections API.

#pragma once

#include <cstddef>

namespace cuco {

/// Dummy allocator type — used as the default Allocator template param.
/// The actual allocation is always hipMalloc; this exists so the template
/// parameter is a real type (not void) and constructor parameters work.
struct dummy_allocator {};

/// Wraps a size value, used as a template parameter pack element.
template <typename T>
struct extent {
  using value_type = T;
  T value;

  __host__ __device__ constexpr extent(T v) : value(v) {}
  __host__ __device__ constexpr operator T() const { return value; }
};

}  // namespace cuco
