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

### 3. `cucascade-rocm` — cuCascade memory reservation port 🔧 In Progress

cuCascade is NVIDIA's GPU memory-reservation and out-of-core data repository system. 63 files in Sirius depend on it. No ROCm port exists.

**Scope:** Port the core subsystem:
- `cucascade::memory::memory_space` — stream + allocator + device context
- `cucascade::memory::memory_reservation_manager` — reservation tracking
- `cucascade::memory::reservation` — reserved capacity handle
- `cucascade::memory::topology_discovery` — multi-GPU topology (rocm-smi/sysfs)
- `cucascade::data::data_batch` / `data_repository` — batch lifecycle

**Approach:** Use hipMM's `rmm::device_async_resource_ref` as the allocator interface, rocThrust/hipCUB for data movement. Match cuCascade's API exactly.

**Status:** 32 headers, of which **26 are real implementations** and **6 still throw at runtime** (→ DuckDB CPU fallback). Verified by grepping every header for the `throw std::runtime_error("cuCascade stub: ...")` marker — the genuine stubs are:

- `memory/small_pinned_host_memory_resource.hpp` — `allocate` / `operator rmm::...` throw
- `memory/numa_region_pinned_host_allocator.hpp` — `allocate` / `operator rmm::...` throw
- `cudf/gpu_data_representation.hpp` — `clone()` throws
- `cudf/host_data_representation.hpp` — `clone()` throws
- `data/disk_data_representation.hpp` — `get_disk_table()` / `clone()` throw
- `data/data_batch.hpp` — `clone_to` / `clone` / `set_data` / `convert_to` throw
- `cudf/builtin_converters.hpp` — `register_builtin_converters` is a no-op (no converters registered → `representation_converter::convert` throws "no converter registered")

The remaining 26 headers are real implementations (not just the 7 previously listed). The full memory-reservation subsystem (`memory_space`, `reservation`, `memory_reservation_manager`, `fixed_size_host_memory_resource`, `reservation_aware_resource_adaptor`, `oom_handling_policy`, `error`), the full topology + event subsystem, the data-repository containers (`data_repository`, `data_repository_manager`), the representation base + converter registry, and the config/common/POD types are all ported. The genuine stubs are concentrated in the data-representation `clone()` methods (deep-copy paths not yet exercised by Sirius) and the two pinned-host allocator variants (NUMA-aware + small-block, which delegate to the real `fixed_size_host_memory_resource` once ported).

See `cucascade-rocm/docs/api-surface.md` for the full API mapping and which scope items remain.

## Build

```bash
# Configure — use hipcc as CXX compiler (needed for -x hip flags)
cmake -B build -S . \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DCMAKE_C_COMPILER=hipcc

# Build tests (requires GPU for runtime)
cmake -B build -S . \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DCMAKE_C_COMPILER=hipcc \
  -DBUILD_TESTS=ON
cmake --build build
ctest --test-dir build

# Install
cmake --install build --prefix /opt/rocm
```

### Verified on real hardware

- **Shim layer:** 0 compile errors on gfx942/ROCm 7.2.1 (hipcc 7.2.53211)
- **cuco-rocm bloom_filter:** 7/7 keys found, FP rate 0.03%, ~3M keys/s
- **cuco-rocm static_set:** 6/6 correct, ~3.4M keys/s
- **Destructor stream-sync:** verified on non-default streams (no use-after-free)
- **hipMM (RMM for HIP):** built + installed to /opt/rocm on real gfx942

### Known build fixes (incorporated into Sirius's build_rocm_deps.sh)

1. **hipcc as C/CXX compiler** — hipMM's .cpp files need `-x hip --offload-arch`
2. **`git config --global http.version HTTP/1.1`** — fixes flaky GitHub clones
3. **`ROCM_AMDGPU_TARGETS=gfx942` env** — avoids rapids_cmake arch mismatch
4. **Pre-clone all 16 CPM deps** — correct branch names from versions.json
5. **Patch get_rmm.cmake** — use find_package(rmm) instead of CPM re-fetch

## License

Apache-2.0 (matching cuco, cuCascade, and Sirius)
