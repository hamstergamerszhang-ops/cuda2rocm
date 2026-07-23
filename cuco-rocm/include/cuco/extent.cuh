/*
 * Copyright 2026, rocm-cuda-compat contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! cuco::extent — size wrapper for bloom_filter and static_set.
//! Matches NVIDIA cuCollections API.

#pragma once

#include <cstddef>
#include <cstdint>

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

/// Dynamic extent marker — used by NVIDIA cuco to indicate a runtime-sized extent.
/// hipDF's groupby/hash code uses cuco::dynamic_extent in template params.
inline constexpr std::size_t dynamic_extent = static_cast<std::size_t>(-1);

/// valid_extent — NVIDIA cuco API used by hipDF's groupby/hash helpers.
/// It's just an extent with an extra Extent parameter (compile-time size or
/// dynamic_extent). We alias it to extent since hipCollections doesn't have
/// compile-time extents.
template <typename SizeType, std::size_t Extent = dynamic_extent>
using valid_extent = extent<SizeType>;

/// make_valid_extent factory — hipDF calls this to create a valid_extent from
/// a runtime size. Since we don't have compile-time extents, just return an
/// extent wrapping the size.
template <typename SizeType>
__host__ constexpr auto make_valid_extent(SizeType size) {
  return extent<SizeType>(size);
}

/// bucket_extent — NVIDIA cuco type for bucket-based storage sizing.
/// hipDF references this in make_valid_extent overloads. We alias to extent.
template <typename SizeType, std::size_t Extent = dynamic_extent>
using bucket_extent = extent<SizeType>;

/// make_bucket_extent factory — creates a bucket_extent from a size.
template <int32_t CGSize, int32_t BucketSize, typename SizeType,
          std::size_t N = dynamic_extent>
__host__ constexpr auto make_bucket_extent(bucket_extent<SizeType, N> const& ext) {
  return extent<SizeType>(ext);
}

/// bucket_storage_ref — NVIDIA cuco internal type used by hipDF's groupby/hash.
/// It's a non-owning view of a bucket storage array. We provide a minimal shim
/// that wraps a pointer + size, matching the API surface hipDF uses.
template <typename KeyT, int32_t BucketSize,
          typename Extent = extent<std::size_t>>
class bucket_storage_ref {
 public:
  using key_type = KeyT;
  using extent_type = Extent;
  using size_type = typename Extent::value_type;

  __host__ __device__ bucket_storage_ref(KeyT* data, Extent ext)
    : data_(data), extent_(ext) {}

  __host__ __device__ KeyT* data() const noexcept { return data_; }
  __host__ __device__ size_type size() const noexcept { return extent_; }
  __host__ __device__ Extent const& extent() const noexcept { return extent_; }

  __host__ __device__ KeyT& operator[](std::size_t i) const { return data_[i]; }

 private:
  KeyT* data_;
  Extent extent_;
};

}  // namespace cuco
