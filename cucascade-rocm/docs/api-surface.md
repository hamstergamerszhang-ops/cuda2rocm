# cucascade-rocm API Surface

The exact cuCascade API subset that Sirius uses, mapped from the source at
`sirius-db/sirius`. This is the minimum surface `cucascade-rocm` must provide.

## Scope

63 source files in Sirius reference `cucascade::` types. The port must provide
real implementations (not stubs) for the core subsystem:

1. **Memory reservation** — `memory_space`, `reservation`, `memory_reservation_manager`
2. **Data repository** — `data_batch`, `data_repository`, `data_repository_manager`
3. **Data representations** — `gpu_table_representation`, `host_data_representation`, `idata_representation`
4. **Topology** — `topology_discovery`, `system_topology_info`
5. **Host resources** — `fixed_size_host_memory_resource`, `reservation_aware_resource_adaptor`
6. **Error handling** — `cucascade_out_of_memory`, `CUCASCADE_CUDA_TRY`
7. **Events** — `cuda_event`, `cuda_event_view`

## Key types and their usage

### memory_space
- Constructor: `(Tier, device_id, allocator, stream_pool)`
- `get_device_id()`, `get_tier()`, `get_id()`, `get_default_allocator()`
- `acquire_stream()` → `rmm::cuda_stream_view`
- `get_memory_resource_as<T>()` → `T*` (reservation_aware_resource_adaptor or fixed_size_host_memory_resource)
- `make_reservation(bytes)`, `make_reservation_or_null(bytes)`, `make_reservation_upto(bytes)`
- `get_max_memory()`, `get_available_memory()`, `get_total_reserved_memory()`

### memory_reservation_manager (virtual base — Sirius subclasses it)
- Constructor: `(vector<memory_space_config>)`
- `get_memory_space(Tier, device_id)` → `memory_space*`
- `get_memory_spaces_for_tier(Tier)` → `span<memory_space* const>`
- `request_reservation(strategy, bytes)` → `unique_ptr<reservation>`

### reservation
- `size()`, `tier()`, `device_id()`, `get_memory_space()`
- `get_memory_resource_as<T>()`, `get_memory_resource_of<TIER>()`

### idata_representation (virtual base — 2 Sirius classes inherit)
- `cast<T>()` (const + non-const) — dynamic_cast
- `clone(stream)` (pure virtual)
- `get_size_in_bytes()`, `get_uncompressed_data_size_in_bytes()` (pure virtual)

### data_batch / read_only_data_batch / mutable_data_batch
- `make(batch_id, data, probe, space)` → `shared_ptr<data_batch>`
- `to_read_only()`, `to_mutable()`, `clone()`, `clone_to<T>()`, `convert_to<T>()`
- `get_data()`, `get_memory_space()`, `get_batch_id()`, `get_state()`

### fixed_size_host_memory_resource
- Nested `multiple_blocks_allocation` with `block` struct (`.data()`, `.size()`, `operator void*()`)
- `allocate_multiple_blocks(bytes, reservation*)` → `unique_ptr<multiple_blocks_allocation>`
- `allocate(stream, bytes, alignment)`, `deallocate(stream, ptr, bytes, alignment)`
- Static constexpr: `default_pool_size`, `default_block_size`, `default_initial_number_pools`

### cucascade_out_of_memory
- Derives from `rmm::out_of_memory`
- Public fields: `error_kind` (MemoryError), `requested_bytes`, `global_usage`, `pool_handle`

## Implementation strategy

### Memory allocation
- Use hipMM's `rmm::device_async_resource_ref` as the allocator interface
- `hipMalloc`/`hipFree` for device memory, `hipMallocHost`/`hipFreeHost` for pinned host
- `hipMemPool*` for stream-ordered memory pools (HIP 7.x supports this)

### Topology
- Use `rocm-smi` / `/sys/class/kfd` for GPU topology discovery
- `hipGetDeviceCount`, `hipDeviceCanAccessPeer` for peer access
- `hipDeviceGetAttribute` for L2 cache, multi-processor count, etc.

### Data movement
- `hipMemcpyAsync` for H2D/D2H
- `hipMemcpyPeerAsync` for peer-to-peer (multi-GPU)
- rocThrust for data-parallel operations

### Events
- `hipEventCreate`, `hipEventRecord`, `hipEventSynchronize`, `hipEventQuery`
- Direct mapping from CUDA event API (already shimmed in cuda_runtime.h)

## Status

A compile-only stub set is provided in this repo at
`cucascade-rocm/include/cucascade/` (consumed by Sirius via FetchContent).
7 of the 32 headers are real implementations (memory_space, stream_pool,
topology_discovery, event, disk_access_limiter, representation_converter,
reservation_manager_configurator); the remaining 25 throw at runtime
(→ DuckDB CPU fallback). This project is replacing those stubs with real
implementations.
