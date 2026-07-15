/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! @c <curand.h> redirect for ROCm.
//! Legacy Sirius code (src/legacy/cuda/operator/cuda_helper.cuh:31) includes
//! @c <curand.h>. ROCm's equivalent is @c <hiprand/hiprand.h> (the bare
//! @c <hiprand.h> path does not exist on stock ROCm 7.2.1 — the header lives
//! under the hiprand/ subdirectory).

#pragma once

#include <hiprand/hiprand.h>
