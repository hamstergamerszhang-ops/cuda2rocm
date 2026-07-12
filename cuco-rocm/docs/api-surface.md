# cuco-rocm API Surface

The exact cuco API subset that Sirius uses, mapped from the source at
`sirius-db/sirius`. This is the minimum surface `cuco-rocm` must provide.

## Headers

```
cuco/bloom_filter.cuh
cuco/bloom_filter_policies.cuh
cuco/hash_functions.cuh
cuco/operator.hpp
cuco/static_set.cuh
```

## Types and their template parameters

### `cuco::bloom_filter<KeyT, Extent, Scope, Policy, Allocator>`

Used in `sirius_dynamic_bloom_filter.cu`:

```cpp
template <class KeyT>
using bloom_filter_for = cuco::bloom_filter<
  KeyT,
  cuco::extent<std::size_t>,
  cuda::thread_scope_device,
  Policy,
  bloom_alloc  // = sirius::rmm_cuco_allocator<cuda::std::byte>
>;
```

**Methods used:**
- Constructor: `bloom_filter(extent, {}, {}, allocator, stream)`
- `add_async(key_begin, key_end, stream)`
- `contains_async(key_begin, key_end, output_begin, stream)`
- `block_extent()` — returns the number of blocks
- `data()` — returns pointer to the bit array
- Static: `words_per_block`, `word_type`, `key_type`

### `cuco::static_set<KeyT, Extent, Scope, Pred, Hash, Allocator>`

Used in `sirius_dynamic_in_list_filter.cu`:

```cpp
template <class KeyT>
using set_type = cuco::static_set<
  KeyT,
  cuco::extent<std::size_t>,
  cuda::thread_scope_device,
  cuda::std::equal_to<KeyT>,
  cuco::double_hashing<CgSize, cuco::default_hash_function<KeyT>>,
  set_alloc  // = sirius::rmm_cuco_allocator<KeyT>
>;
```

**Methods used:**
- Constructor: `static_set(extent, empty_key, {}, {}, {}, allocator, stream)`
- `insert_async(key_begin, key_end, stream)`
- `ref(cuco::contains)` — returns a reference object for probing
- `capacity()` — returns the set capacity
- `data()` — returns pointer to the slot array

### Supporting types

| Type | Purpose | Notes |
|---|---|---|
| `cuco::extent<std::size_t>` | Size wrapper | Constructed from `std::size_t` |
| `cuco::empty_key<KeyT>` | Sentinel for empty slots | Constructed from `KeyT` |
| `cuco::arrow_filter_policy<KeyT>` | Bloom filter hashing policy | Has `max_filter_blocks` static |
| `cuco::default_filter_policy<Hash, WordT, WordsPerBlock>` | Standard Bloom policy | |
| `cuco::xxhash_64<KeyT>` | xxHash 64-bit | Used as the hash function |
| `cuco::double_hashing<CgSize, Hash>` | Double hashing probe strategy | |
| `cuco::default_hash_function<KeyT>` | Default hash | Used inside double_hashing |
| `cuco::contains` | Operator tag for `ref()` | Empty struct |

## Implementation strategy

### Primitives available on ROCm

- `rocprim::warp_shuffle` / `hipcub::ShuffleIndex` — for warp-level coordination
- `rocprim::block_scan` / `hipcub::BlockScan` — for block-level reductions
- `hipMalloc` / `hipFree` — for device memory (via RMM/hipMM allocator)
- `hipAtomicAdd` / `hipAtomicOr` — for atomic bit-set in Bloom filter
- `hipDeviceSynchronize` / streams — for async operations

### Bloom filter implementation outline

```
class bloom_filter {
  word_type* bits_;        // device memory, allocated via Allocator
  extent block_extent_;    // number of blocks
  Policy policy_;          // hashing policy (k hash functions)
  
  add_async(keys, stream):
    kernel<<<grid, block, 0, stream>>>(bits_, policy_, keys, n);
    // Each thread hashes its key, sets k bits via atomicOr
    
  contains_async(keys, output, stream):
    kernel<<<grid, block, 0, stream>>>(bits_, policy_, keys, output, n);
    // Each thread hashes its key, checks k bits, writes bool
}
```

### static_set implementation outline

```
class static_set {
  slot_type* slots_;       // device memory
  extent capacity_;
  key_type empty_key_;     // sentinel
  Hash hash_;              // double_hashing
  
  insert_async(keys, stream):
    kernel<<<grid, block, 0, stream>>>(slots_, hash_, empty_key_, keys, n);
    // Each thread probes open-addressing table, atomicCAS to insert
    
  ref(contains):
    returns a view object with operator() that probes
}
```

## xxHash on HIP

`cuco::xxhash_64<KeyT>` needs a HIP-compatible xxHash implementation.
Options:
1. Port the xxHash C reference implementation to device code (straightforward)
2. Use ROCm's `hiprand` hash functions (different algorithm, not compatible)
3. Use a simple multiply-shift hash (breaks compatibility with cuco's behavior)

Recommend option 1 — xxHash is ~50 lines of C, ports cleanly to `__device__`.
