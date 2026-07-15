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
  Tier tier() const { return space_ ? space_->get_tier() : Tier::HOST; }
  int32_t device_id() const { return space_ ? space_->get_device_id() : 0; }
  memory_space const& get_memory_space() const {
    if (!space_) { throw std::runtime_error("cuCascade: reservation has no memory_space"); }
    return *space_;
  }

  void grow_by(std::size_t n) { if (arena_) arena_->grow_by(n); }
  void shrink_to_fit() { if (arena_) arena_->shrink_to_fit(); }

  // Delegate to the owning memory_space. memory_space is only forward-declared
  // here (its full definition is in memory_space.hpp, which includes this
  // file), so these templates are defined inline but rely on two-phase name
  // lookup: the complete type is required only at instantiation (i.e. at the
  // call site, where memory_space.hpp is already in scope), not at definition.
  // This mirrors the real cuCascade: a reservation is a handle into its space's
  // resource registry.
  template <typename T>
  T* get_memory_resource_as() {
    return space_ ? space_->template get_memory_resource_as<T>() : nullptr;
  }
  template <typename T>
  T const* get_memory_resource_as() const {
    return space_ ? space_->template get_memory_resource_as<T>() : nullptr;
  }

  template <Tier TIER>
  typename tier_memory_resource_trait<TIER>::type* get_memory_resource_of() {
    return space_ ? space_->template get_memory_resource_of<TIER>() : nullptr;
  }
  template <Tier TIER>
  typename tier_memory_resource_trait<TIER>::type const* get_memory_resource_of() const {
    return space_ ? space_->template get_memory_resource_of<TIER>() : nullptr;
  }

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
