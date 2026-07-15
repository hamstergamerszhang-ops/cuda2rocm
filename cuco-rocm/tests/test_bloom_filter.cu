/*
 * Copyright 2026, rocm-cuda-compat contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! Test: cuco::bloom_filter compiles and basic insert/probe works.

#include <cuco/bloom_filter.cuh>
#include <cuco/extent.cuh>
#include <cuco/bloom_filter_policies.cuh>

#include <hip/hip_runtime.h>
#include <cassert>
#include <cstdint>
#include <cstdio>

// Check HIP runtime calls and abort with a clear message on failure, so an
// allocation/copy failure surfaces as a test error instead of a nullptr
// dereference inside a kernel.
#define HIP_CHECK(call) do { \
    hipError_t _err = (call); \
    if (_err != hipSuccess) { \
      printf("HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
      return 1; \
    } \
  } while (0)

int main() {
  using key_t = int32_t;
  using filter_t = cuco::bloom_filter<key_t>;

  // Create a bloom filter with 1024 blocks
  filter_t bf(cuco::extent<std::size_t>{1024});
  printf("Bloom filter created: %zu blocks, %zu words/block\n",
         bf.block_extent(), filter_t::words_per_block);

  // Insert some keys on host, copy to device, add
  key_t h_keys[] = {1, 2, 3, 42, 100, 200, 300};
  key_t* d_keys;
  HIP_CHECK(hipMalloc(&d_keys, sizeof(h_keys)));
  HIP_CHECK(hipMemcpy(d_keys, h_keys, sizeof(h_keys), hipMemcpyHostToDevice));

  bf.add_async(d_keys, d_keys + 7, 0);
  hipStreamSynchronize(0);

  // Probe
  bool* d_results;
  HIP_CHECK(hipMalloc(&d_results, sizeof(bool) * 7));
  bf.contains_async(d_keys, d_keys + 7, d_results, 0);
  hipStreamSynchronize(0);

  bool h_results[7];
  HIP_CHECK(hipMemcpy(h_results, d_results, sizeof(h_results), hipMemcpyDeviceToHost));

  int pass = 0;
  for (int i = 0; i < 7; i++) {
    printf("  key %d: %s\n", h_keys[i], h_results[i] ? "FOUND" : "MISSING");
    if (h_results[i]) pass++;
  }

  // Test a key NOT in the set
  key_t h_miss[] = {999};
  key_t* d_miss;
  HIP_CHECK(hipMalloc(&d_miss, sizeof(h_miss)));
  HIP_CHECK(hipMemcpy(d_miss, h_miss, sizeof(h_miss), hipMemcpyHostToDevice));
  bool* d_miss_res;
  HIP_CHECK(hipMalloc(&d_miss_res, sizeof(bool)));
  bf.contains_async(d_miss, d_miss + 1, d_miss_res, 0);
  hipStreamSynchronize(0);
  bool miss_res;
  HIP_CHECK(hipMemcpy(&miss_res, d_miss_res, sizeof(bool), hipMemcpyDeviceToHost));
  printf("  key 999 (not inserted): %s\n", miss_res ? "FOUND (false positive!)" : "MISSING (correct)");

  hipFree(d_keys);
  hipFree(d_results);
  hipFree(d_miss);
  hipFree(d_miss_res);

  printf("Result: %d/7 keys found, miss key %s\n", pass, miss_res ? "false positive" : "correctly missing");
  assert(pass == 7 && "All inserted keys must be found");
  assert(!miss_res && "Non-inserted key must not be found (no false positive)");
  printf("TEST PASSED\n");
  return (pass == 7 && !miss_res) ? 0 : 1;
}
