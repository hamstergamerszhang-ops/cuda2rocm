/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade data_batch — ROCm real implementation.
//! The most-used cuCascade type. data_batch / read_only_data_batch / mutable_data_batch.
//! clone() delegates to idata_representation::clone() (implemented for host/gpu
//! representations in their respective headers). clone_to() + convert_to() use
//! the representation_converter_registry to convert between representations
//! (e.g. gpu -> host for spilling). set_data() is a simple ownership swap.

#pragma once
#include "cucascade/data/common.hpp"
#include "cucascade/data/representation_converter.hpp"
#include "cucascade/memory/common.hpp"
#include <rmm/cuda_stream.hpp>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

namespace cucascade::memory { class memory_space; }

namespace cucascade {

/// Read-only view of a data batch (copyable, holds a shared reference).
class read_only_data_batch {
 public:
  read_only_data_batch() = default;
  read_only_data_batch(read_only_data_batch const&) = default;
  read_only_data_batch& operator=(read_only_data_batch const&) = default;
  read_only_data_batch(read_only_data_batch&&) noexcept = default;
  read_only_data_batch& operator=(read_only_data_batch&&) noexcept = default;

  read_only_data_batch(std::shared_ptr<idata_representation> data,
                       memory::memory_space* space, uint64_t batch_id)
    : data_(std::move(data)), space_(space), batch_id_(batch_id) {}

  idata_representation const* get_data() const { return data_.get(); }
  std::shared_ptr<idata_representation> const& get_shared_data() const { return data_; }
  memory::memory_space* get_memory_space() const { return space_; }
  memory::Tier get_current_tier() const { return memory::Tier::GPU; }
  uint64_t get_batch_id() const { return batch_id_; }
  cudaEvent_t get_writer_event() const { return nullptr; }

  /// Convert this batch to a target representation type via the converter
  /// registry, returning a new data_batch with the converted data. Used by
  /// Sirius to spill GPU data to host (gpu_table_representation ->
  /// host_data_representation) when VRAM is tight. The converter is looked up
  /// by (source_type, target_type) in the registry; if no converter is
  /// registered, convert() throws (the registry's own error, not a stub).
  template <typename TargetRepr>
  std::shared_ptr<class data_batch> clone_to(representation_converter_registry& registry,
                                             uint64_t new_batch_id,
                                             memory::memory_space const* target_space,
                                             rmm::cuda_stream_view stream,
                                             std::unique_ptr<idata_batch_probe> /*probe*/ = {}) {
    if (!data_) {
      throw std::runtime_error("cuCascade: read_only_data_batch::clone_to on null data");
    }
    auto converted = registry.convert<TargetRepr>(*data_, target_space, stream);
    auto* target_space_mut = const_cast<memory::memory_space*>(target_space);
    return data_batch::make(new_batch_id, std::move(converted), {},
                            target_space_mut);
  }

  /// Deep-copy this batch's data representation (same type, same tier) into a
  /// new data_batch with a new batch_id. Delegates to
  /// idata_representation::clone(), which is implemented per representation
  /// type (host_data_representation::clone does hipMemcpyAsync; gpu_table does
  /// cudf::copy). The clone is independent — mutations to one don't affect
  /// the other.
  std::shared_ptr<class data_batch> clone(uint64_t new_batch_id,
                                          rmm::cuda_stream_view stream,
                                          std::unique_ptr<idata_batch_probe> /*probe*/ = {}) {
    if (!data_) {
      throw std::runtime_error("cuCascade: read_only_data_batch::clone on null data");
    }
    auto cloned = data_->clone(stream);
    return data_batch::make(new_batch_id, std::move(cloned), {}, space_);
  }

 protected:
  std::shared_ptr<idata_representation> data_;
  memory::memory_space* space_{nullptr};
  uint64_t batch_id_{0};
};

/// Mutable view of a data batch (move-only, holds exclusive access).
class mutable_data_batch {
 public:
  mutable_data_batch() = default;
  mutable_data_batch(mutable_data_batch&&) noexcept = default;
  mutable_data_batch& operator=(mutable_data_batch&&) noexcept = default;

  mutable_data_batch(std::shared_ptr<idata_representation> data,
                     memory::memory_space* space, uint64_t batch_id)
    : data_(std::move(data)), space_(space), batch_id_(batch_id) {}

  idata_representation const* get_data() const { return data_.get(); }
  std::shared_ptr<idata_representation> const& get_shared_data() const { return data_; }
  memory::memory_space* get_memory_space() const { return space_; }
  uint64_t get_batch_id() const { return batch_id_; }

  /// Replace the held data representation with a new one. The old data is
  /// released (its refcount drops; if this was the last reference, it's
  /// destroyed and its buffer freed). Used by Sirius when an in-place
  /// transformation produces a new representation (e.g. re-filtering a column).
  void set_data(std::unique_ptr<idata_representation> new_data) {
    data_ = std::move(new_data);
  }

  void rebind_stream(rmm::cuda_stream_view) {}

  /// Convert the batch's data to a target representation type. Returns a
  /// unique_ptr to the converted representation (the caller owns it). Used by
  /// Sirius for one-off conversions that don't need a full data_batch wrapper.
  template <typename TargetRepr>
  std::unique_ptr<TargetRepr> convert_to(representation_converter_registry& registry,
                                         memory::memory_space const* target_space,
                                         rmm::cuda_stream_view stream) {
    if (!data_) {
      throw std::runtime_error("cuCascade: mutable_data_batch::convert_to on null data");
    }
    return registry.convert<TargetRepr>(*data_, target_space, stream);
  }

  /// Convert + wrap in a new data_batch (same as read_only_data_batch::clone_to
  /// but from the mutable view). The mutable batch's data is NOT consumed —
  /// the converter produces a new representation from the existing one.
  template <typename TargetRepr>
  std::shared_ptr<class data_batch> clone_to(representation_converter_registry& registry,
                                             uint64_t new_batch_id,
                                             memory::memory_space const* target_space,
                                             rmm::cuda_stream_view stream,
                                             std::unique_ptr<idata_batch_probe> = {}) {
    if (!data_) {
      throw std::runtime_error("cuCascade: mutable_data_batch::clone_to on null data");
    }
    auto converted = registry.convert<TargetRepr>(*data_, target_space, stream);
    auto* target_space_mut = const_cast<memory::memory_space*>(target_space);
    return data_batch::make(new_batch_id, std::move(converted), {},
                            target_space_mut);
  }

 private:
  std::shared_ptr<idata_representation> data_;
  memory::memory_space* space_{nullptr};
  uint64_t batch_id_{0};
};

/// A data_batch owns a data representation and hands out shared read-only or
/// mutable views of it. Converting between read-only and mutable views must
/// preserve shared ownership of the underlying idata_representation; callers
/// receive a new view that references the same object.
class data_batch {
 public:
  static std::shared_ptr<data_batch> make(uint64_t batch_id,
                                          std::unique_ptr<idata_representation> data,
                                          std::unique_ptr<idata_batch_probe> probe = {},
                                          memory::memory_space* space = nullptr) {
    return std::shared_ptr<data_batch>(new data_batch(batch_id, std::move(data), std::move(probe), space));
  }

  batch_state get_state() const { return state_; }
  uint64_t get_batch_id() const { return batch_id_; }

  read_only_data_batch to_read_only() {
    if (state_ != batch_state::mutable_locked && state_ != batch_state::idle) {
      throw std::runtime_error("cuCascade: to_read_only from invalid state");
    }
    state_ = batch_state::read_only;
    return read_only_data_batch(data_, space_, batch_id_);
  }
  mutable_data_batch to_mutable() {
    if (state_ != batch_state::read_only && state_ != batch_state::idle) {
      throw std::runtime_error("cuCascade: to_mutable from invalid state");
    }
    state_ = batch_state::mutable_locked;
    return mutable_data_batch(data_, space_, batch_id_);
  }

  bool try_to_read_only() {
    if (state_ != batch_state::mutable_locked && state_ != batch_state::idle) return false;
    state_ = batch_state::read_only;
    return true;
  }
  bool try_to_mutable() {
    if (state_ != batch_state::read_only && state_ != batch_state::idle) return false;
    state_ = batch_state::mutable_locked;
    return true;
  }

  static read_only_data_batch to_idle(read_only_data_batch&& ro) {
    return std::move(ro);  // state transitions handled by caller
  }
  static mutable_data_batch to_idle(mutable_data_batch&& mut) {
    return std::move(mut);
  }
  static mutable_data_batch readonly_to_mutable(read_only_data_batch&& ro) {
    auto space = ro.get_memory_space();
    auto id = ro.get_batch_id();
    return mutable_data_batch(ro.get_shared_data(), space, id);
  }
  static read_only_data_batch mutable_to_readonly(mutable_data_batch&& mut) {
    auto space = mut.get_memory_space();
    auto id = mut.get_batch_id();
    return read_only_data_batch(mut.get_shared_data(), space, id);
  }

  void subscribe() { ++subscriber_count_; }
  void unsubscribe() { if (subscriber_count_ > 0) --subscriber_count_; }
  std::size_t get_subscriber_count() const { return subscriber_count_; }

 private:
  data_batch(uint64_t batch_id, std::unique_ptr<idata_representation> data,
             std::unique_ptr<idata_batch_probe> /*probe*/, memory::memory_space* space)
    : batch_id_(batch_id), data_(std::move(data)), space_(space) {}

  uint64_t batch_id_{0};
  batch_state state_{batch_state::idle};
  std::shared_ptr<idata_representation> data_;
  memory::memory_space* space_{nullptr};
  std::size_t subscriber_count_{0};
};

}  // namespace cucascade
