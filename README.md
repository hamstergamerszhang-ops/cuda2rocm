# rocm-cuda-compat

CUDA→ROCm compatibility tools for porting NVIDIA-only GPU libraries to AMD HIP.

## Projects

### 1. `cuda-compat-shims` — Reusable CUDA→HIP shim headers ✅ Verified

The shim layer developed for the [Sirius ROCm port](https://github.com/hamstergamerszhang-ops/sirius/tree/feat/rocm-port), extracted as a standalone header-only library usable by any CUDA→ROCm porting effort:

- `cuda_runtime.h` — 68 `cuda*`→`hip*` macro aliases + 6 type aliases (verified on gfx942)
- `cub/` → hipCUB redirect + `namespace cub = hipcub` + `warp_threads = 64`
- `cuda/std/*` — CCCL redirects with `#include_next` (defers to hipDF's CCCL when installed)
- `cuda/functional` — `cuda::minimum`/`cuda::maximum` functors
- `cuda/stream_ref`, `cuda/cmath`, `cuda/__memory/aligned_size.h`, `cuda/atomic`, `cuda/memory_resource`
- `cooperative_groups` memcpy_async/wait serial fallback
- `nvtx3` → `roctx` shim
- `curand.h` → `hiprand.h` redirect

**Status:** ✅ Verified on real gfx942/ROCm 7.2.1 (0 compile errors, 6 benign warnings).

**Usage:**
```cmake
find_package(rocm-cuda-compat CONFIG REQUIRED)
target_link_libraries(my-target PRIVATE rocm-cuda-compat::cuda-compat-shims)
```

### 2. `cuco-rocm` — cuCollections port for ROCm/HIP 🔧 In Progress

cuCollections (cuco) is NVIDIA's GPU hash-map and Bloom-filter library, used by Sirius, RAPIDS, and the broader CUDA ecosystem. No ROCm port exists.

**Scope:** Port the subset of cuco that Sirius uses:
- `cuco::bloom_filter` (blocked Bloom filter) — ✅ Drafted, needs testing
- `cuco::static_set` (open-addressing hash set) — ✅ Drafted, needs testing
- `cuco::xxhash_64` (device-compatible xxHash) — ✅ Implemented
- Supporting types: `cuco::extent`, `cuco::empty_key`, `cuco::arrow_filter_policy`,
  `cuco::default_filter_policy`, `cuco::double_hashing`, `cuco::default_hash_function`,
  `cuco::contains`

**Approach:** Use rocPRIM/hipCUB primitives (warp shuffles, atomicCAS) as building blocks, matching cuco's API surface exactly so downstream code compiles without changes. Target: `namespace cuco`, same headers, same templates.

**Status:** Drafted (6 headers). Needs compile + runtime testing on gfx942.

### 3. `cucascade-rocm` — cuCascade memory reservation port 📋 Planned

cuCascade is NVIDIA's GPU memory-reservation and out-of-core data repository system. 63 files in Sirius depend on it. No ROCm port exists.

**Scope:** Port the core subsystem:
- `cucascade::memory::memory_space` — stream + allocator + device context
- `cucascade::memory::memory_reservation_manager` — reservation tracking
- `cucascade::memory::reservation` — reserved capacity handle
- `cucascade::memory::topology_discovery` — multi-GPU topology (rocm-smi/sysfs)
- `cucascade::data::data_batch` / `data_repository` — batch lifecycle

**Approach:** Use hipMM's `rmm::device_async_resource_ref` as the allocator interface, rocThrust/hipCUB for data movement. Match cuCascade's API exactly.

**Status:** Compile-only stubs exist (32 headers, verified — throw at runtime → DuckDB CPU fallback). Real implementation is planned, staged:
1. Memory reservation subsystem (`memory_space`, `reservation`, `memory_reservation_manager`)
2. Data repository (`data_batch`, `data_repository`, `data_repository_manager`)
3. Topology discovery + disk representations

See `cucascade-rocm/docs/api-surface.md` for the full API mapping.

## Build

```bash
# Configure
cmake -B build -S . -DCMAKE_HIP_ARCHITECTURES=gfx942

# Build tests (requires GPU for runtime)
cmake -B build -S . -DCMAKE_HIP_ARCHITECTURES=gfx942 -DBUILD_TESTS=ON
cmake --build build
ctest --test-dir build

# Install
cmake --install build --prefix /opt/rocm
```

## License

Apache-2.0 (matching cuco, cuCascade, and Sirius)
