/*
 * Copyright 2026, rocm-cuda-compat contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! cuco::operator — operator tags for static_set::ref().
//! Matches NVIDIA cuCollections API.

#pragma once

namespace cuco {

/// Tag for contains queries: `set.ref(cuco::contains)`.
struct contains_t {};
inline constexpr contains_t contains{};

}  // namespace cuco
