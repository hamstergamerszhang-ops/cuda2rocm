/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! cuCascade memory reservation — ROCm stub.

#pragma once

#include "cucascade/memory/common.hpp"
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>

namespace cucascade::memory {

class memory_space;

/// Abstract base for a reserved memory arena.
class reserved_arena {
 public:
  virtual ~reserved_arena() = default;
  virtual std::size_t size() const = 0;
  virtual void grow_by(std::size_t) {}
  virtual void shrink_to_fit() {}
};

/// Reservation limit policies.
class reservation_limit_policy {
 public:
  virtual ~reservation_limit_policy() = default;
  virtual bool can_reserve(std::size_t /*requested*/, std::size_t /*current*/) const { return true; }
};

class ignore_reservation_limit_policy : public reservation_limit_policy {};
class fail_reservation_limit_policy : public reservation_limit_policy {
 public:
  bool can_reserve(std::size_t, std::size_t) const override { return false; }
};
class increase_reservation_limit_policy : public reservation_limit_policy {};

/// A memory reservation — represents reserved capacity on a memory space.
/// The owning space may be null for host-only reservations that are tracked by
/// a host memory resource instead of a memory_space.
class reservation {
 public:
  using release_callback = std::function<void()>;

  explicit reservation(memory_space* space, std::unique_ptr<reserved_arena> arena,
                       release_callback on_release = {})
    : space_(space), arena_(std::move(arena)), on_release_(std::move(on_release)) {}

  ~reservation() {
    if (on_release_) on_release_();
  }

  std::size_t size() const {
    return arena_ ? arena_->size() : 0;
  }
  // These 5 methods delegate to the owning memory_space, which is only
  // forward-declared here (its full definition is in memory_space.hpp, which
  // includes this file). They are DECLARED here but DEFINED out-of-line at the
  // bottom of memory_space.hpp (after memory_space is complete). The old code
  // defined them inline and relied on two-phase-lookup leniency to defer the
  // member access — gcc/hipcc accept that, but clang rejects it ("member access
  // into incomplete type"). Moving the definitions is the standard C++ idiom
  // for this circular dependency and makes the code portable.
  Tier tier() const;
  int32_t device_id() const;
  memory_space const& get_memory_space() const;

  void grow_by(std::size_t n) { if (arena_) arena_->grow_by(n); }
  void shrink_to_fit() { if (arena_) arena_->shrink_to_fit(); }

  /// Read-only access to the underlying arena. Used by host memory resources
  /// to look up the per-reservation allocation tracker in
  /// allocate_multiple_blocks(reservation*) — the tracker is keyed by the
  /// arena pointer, mirroring the real cuCascade's chunked_reserved_area map.
  reserved_arena* arena() const noexcept { return arena_.get(); }

  // Template overloads that delegate to memory_space — also defined out-of-line
  // at the bottom of memory_space.hpp.
  template <typename T>
  T* get_memory_resource_as();
  template <typename T>
  T const* get_memory_resource_as() const;

  template <Tier TIER>
  typename tier_memory_resource_trait<TIER>::type* get_memory_resource_of();
  template <Tier TIER>
  typename tier_memory_resource_trait<TIER>::type const* get_memory_resource_of() const;

  static std::unique_ptr<reservation> create(memory_space& space,
                                             std::unique_ptr<reserved_arena> arena,
                                             release_callback on_release = {}) {
    return std::make_unique<reservation>(&space, std::move(arena), std::move(on_release));
  }

  static std::unique_ptr<reservation> create(std::unique_ptr<reserved_arena> arena,
                                             release_callback on_release = {}) {
    return std::make_unique<reservation>(nullptr, std::move(arena), std::move(on_release));
  }

 private:
  memory_space* space_;
  std::unique_ptr<reserved_arena> arena_;
  release_callback on_release_;
};

}  // namespace cucascade::memory
