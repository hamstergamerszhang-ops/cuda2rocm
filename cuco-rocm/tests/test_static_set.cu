/*
 * Copyright 2026, rocm-cuda-compat contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! Test: cuco::static_set compiles and basic insert/contains works.

#include <cuco/static_set.cuh>
#include <cuco/extent.cuh>
#include <cuco/operator.hpp>

#include <hip/hip_runtime.h>
#include <cassert>
#include <climits>
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

using key_t = int32_t;
using set_t = cuco::static_set<key_t>;
using set_ref_t = cuco::set_ref<key_t, set_t::hasher_type>;

__global__ void probe_kernel(key_t* keys, bool* results, set_ref_t ref, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    results[idx] = ref(keys[idx]);
  }
}

int main() {
  // Create a set with capacity 256, empty key = INT32_MIN
  set_t ss(cuco::extent<std::size_t>{256},
           cuco::empty_key<key_t>{INT32_MIN});
  printf("Static set created: capacity %zu\n", ss.capacity());

  // Insert some keys
  key_t h_keys[] = {10, 20, 30, 40, 50};
  key_t* d_keys;
  HIP_CHECK(hipMalloc(&d_keys, sizeof(h_keys)));
  HIP_CHECK(hipMemcpy(d_keys, h_keys, sizeof(h_keys), hipMemcpyHostToDevice));

  ss.insert_async(d_keys, d_keys + 5, 0);
  hipStreamSynchronize(0);

  // Probe
  set_ref_t ref = ss.ref(cuco::contains);

  key_t h_probe[] = {10, 20, 30, 40, 50, 999};
  key_t* d_probe;
  bool* d_results;
  HIP_CHECK(hipMalloc(&d_probe, sizeof(h_probe)));
  HIP_CHECK(hipMalloc(&d_results, sizeof(bool) * 6));
  HIP_CHECK(hipMemcpy(d_probe, h_probe, sizeof(h_probe), hipMemcpyHostToDevice));

  probe_kernel<<<1, 6>>>(d_probe, d_results, ref, 6);
  hipStreamSynchronize(0);

  bool h_results[6];
  HIP_CHECK(hipMemcpy(h_results, d_results, sizeof(h_results), hipMemcpyDeviceToHost));

  int pass = 0;
  for (int i = 0; i < 6; i++) {
    printf("  key %d: %s\n", h_probe[i], h_results[i] ? "FOUND" : "MISSING");
    if (i < 5 && h_results[i]) pass++;
    if (i == 5 && !h_results[i]) pass++;
  }

  hipFree(d_keys);
  hipFree(d_probe);
  hipFree(d_results);

  printf("Result: %d/6 correct\n", pass);
  assert(pass == 6 && "All 5 inserted keys found + 1 non-inserted key correctly missing");
  printf("TEST PASSED\n");
  return (pass == 6) ? 0 : 1;
}
