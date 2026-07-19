/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade topology_discovery — ROCm real implementation.
//!
//! Queries GPU topology via HIP runtime + Linux sysfs. No NVML or hwloc
//! dependency — uses /sys/class/drm/ and /sys/class/kfd/ for NUMA node,
//! CPU affinity, and interconnect quality detection.
//!
//! Field names match what src/sirius_context.cpp and src/include/sirius_config.hpp
//! actually access (id, name, numa_node, pci_bus_id, num_gpus, num_numa_nodes, hostname).

#pragma once
#include <cuda_runtime.h>  // shim → hip runtime
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <unistd.h>  // gethostname

namespace cucascade::memory {

enum class StorageDriveType { NVME, SATA_SSD, HDD, UNKNOWN, SIZE };
enum class NetworkDeviceVerification { EXISTS_ACTIVE_IP, EXISTS, NONE, SIZE };

/// Interconnect quality between two GPUs.
/// DIRECT = same XGMI link (coherent, fastest)
/// MULTIHOP = different PCIe bridge / cross-NUMA (slower peer DMA)
/// UNKNOWN = couldn't determine
enum class PciePathType { DIRECT, MULTIHOP, UNKNOWN, SIZE };

struct gpu_topology_info {
  int32_t id{0};
  std::string name;
  int32_t numa_node{-1};
  std::string pci_bus_id;
  std::size_t total_memory{0};
  std::vector<int32_t> peer_devices;
  /// CPU cores affinity (from /sys/class/drm/cardN/device/local_cpulist)
  std::string cpu_affinity_list;
  /// Interconnect type to each peer GPU: "XGMI" or "PCIe"
  std::vector<std::string> peer_interconnect_types;
};

struct network_device_info {
  std::string name;
  std::string ip_address;
  bool is_active{false};
};

struct storage_device_info {
  std::string mount_point;
  StorageDriveType drive_type{StorageDriveType::UNKNOWN};
  std::size_t capacity{0};
};

struct system_topology_info {
  std::size_t num_gpus{0};
  std::size_t num_numa_nodes{0};
  std::string hostname;
  std::vector<gpu_topology_info> gpus;
  std::vector<network_device_info> network_devices;
  std::vector<storage_device_info> storage_devices;
};

class topology_discovery {
 public:
  bool discover(NetworkDeviceVerification /*verify*/ = NetworkDeviceVerification::EXISTS_ACTIVE_IP) {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count <= 0) {
      return false;
    }
    topology_.num_gpus = static_cast<std::size_t>(device_count);
    topology_.gpus.clear();
    topology_.gpus.reserve(device_count);

    std::unordered_set<int32_t> numa_nodes_seen;

    for (int i = 0; i < device_count; ++i) {
      gpu_topology_info gpu;
      gpu.id = i;

      // Device name + memory
      hipDeviceProp_t prop;
      if (hipGetDeviceProperties(&prop, i) == hipSuccess) {
        gpu.name = prop.name;
        gpu.total_memory = prop.totalGlobalMem;
      }

      // NUMA node via /sys/class/drm/cardN/device/numa_node
      gpu.numa_node = read_sysfs_numa_node(i);
      if (gpu.numa_node >= 0) {
        numa_nodes_seen.insert(gpu.numa_node);
      }

      // PCI bus ID
      char pci_bus_id[32] = {0};
      if (hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), i) == hipSuccess) {
        gpu.pci_bus_id = pci_bus_id;
      }

      // CPU affinity via /sys/class/drm/cardN/device/local_cpulist
      gpu.cpu_affinity_list = read_sysfs_cpulist(i);

      // Peer access + interconnect type
      for (int j = 0; j < device_count; ++j) {
        if (j == i) continue;
        int can_access = 0;
        if (hipDeviceCanAccessPeer(&can_access, i, j) == hipSuccess && can_access) {
          gpu.peer_devices.push_back(j);
          // Detect interconnect type: XGMI (coherent) vs PCIe
          gpu.peer_interconnect_types.push_back(detect_interconnect_type(i, j));
        }
      }

      topology_.gpus.push_back(std::move(gpu));
    }

    topology_.num_numa_nodes = std::max(numa_nodes_seen.size(), std::size_t{1});

    char hostname[256] = {0};
    gethostname(hostname, sizeof(hostname) - 1);
    topology_.hostname = hostname;

    discovered_ = true;
    return true;
  }

  bool is_discovered() const { return discovered_; }
  system_topology_info const& get_topology() const { return topology_; }

 private:
  system_topology_info topology_;
  bool discovered_{false};

  /// Read NUMA node from /sys/class/drm/cardN/device/numa_node.
  /// Returns -1 if unavailable.
  static int32_t read_sysfs_numa_node(int gpu_id) {
    // Try renderDNN first (more reliable on AMD)
    for (int render = 128; render < 200; ++render) {
      std::string path = "/sys/class/drm/renderD" + std::to_string(render) +
                         "/device/numa_node";
      // Check if this renderD belongs to our GPU by matching PCI bus ID
      if (matches_gpu_pci(render, gpu_id)) {
        std::ifstream f(path);
        if (f) {
          int32_t node = -1;
          if (f >> node) return node;
          return -1;  // parse failure — don't return 0 (valid NUMA node)
        }
      }
    }
    // Fallback: scan cardN devices (0..31) by PCI bus match. The old code
    // used the HIP device index directly as cardN, but HIP enumeration order
    // need not match the drm cardN probe order — so the fallback could read
    // the wrong GPU's numa_node and attribute it to gpu_id. Scan by PCI slot
    // instead, the same way the renderD path does.
    for (int card = 0; card < 32; ++card) {
      if (!matches_card_pci(card, gpu_id)) continue;
      std::string card_path = "/sys/class/drm/card" + std::to_string(card) +
                              "/device/numa_node";
      std::ifstream f(card_path);
      if (f) {
        int32_t node = -1;
        if (f >> node) return node;
        return -1;  // parse failure — don't return 0 (valid NUMA node)
      }
    }
    return -1;
  }

  /// Read CPU affinity list from /sys/class/drm/cardN/device/local_cpulist
  static std::string read_sysfs_cpulist(int gpu_id) {
    for (int render = 128; render < 200; ++render) {
      std::string path = "/sys/class/drm/renderD" + std::to_string(render) +
                         "/device/local_cpulist";
      if (matches_gpu_pci(render, gpu_id)) {
        std::ifstream f(path);
        if (f) {
          std::string cpus;
          std::getline(f, cpus);
          return cpus;
        }
      }
    }
    // Fallback: scan cardN devices by PCI match (same rationale as
    // read_sysfs_numa_node — HIP index ≠ drm cardN).
    for (int card = 0; card < 32; ++card) {
      if (!matches_card_pci(card, gpu_id)) continue;
      std::string card_path = "/sys/class/drm/card" + std::to_string(card) +
                              "/device/local_cpulist";
      std::ifstream f(card_path);
      if (f) {
        std::string cpus;
        std::getline(f, cpus);
        return cpus;
      }
    }
    return "";
  }

  /// Check if a renderD device matches the given GPU's PCI bus ID.
  static bool matches_gpu_pci(int render_id, int gpu_id) {
    char gpu_pci[32] = {0};
    if (hipDeviceGetPCIBusId(gpu_pci, sizeof(gpu_pci), gpu_id) != hipSuccess) {
      return false;
    }
    // Read the renderD's PCI slot
    std::string path = "/sys/class/drm/renderD" + std::to_string(render_id) +
                       "/device/uevent";
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    std::string render_pci;
    while (std::getline(f, line)) {
      if (line.rfind("PCI_SLOT_NAME=", 0) == 0) {
        render_pci = line.substr(14);
        break;
      }
    }
    if (render_pci.empty()) return false;
    // Normalize: GPU returns "0000:XX:YY.Z", sysfs may use same or "0000:XX:YY.Z"
    // Just compare the bus:device portion
    return pci_buses_match(gpu_pci, render_pci.c_str());
  }

  /// Same as matches_gpu_pci but for the cardN (not renderDNN) sysfs path.
  /// Used by the fallback scan in read_sysfs_numa_node / read_sysfs_cpulist.
  static bool matches_card_pci(int card_id, int gpu_id) {
    char gpu_pci[32] = {0};
    if (hipDeviceGetPCIBusId(gpu_pci, sizeof(gpu_pci), gpu_id) != hipSuccess) {
      return false;
    }
    std::string path = "/sys/class/drm/card" + std::to_string(card_id) +
                       "/device/uevent";
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    std::string card_pci;
    while (std::getline(f, line)) {
      if (line.rfind("PCI_SLOT_NAME=", 0) == 0) {
        card_pci = line.substr(14);
        break;
      }
    }
    if (card_pci.empty()) return false;
    return pci_buses_match(gpu_pci, card_pci.c_str());
  }

  /// Compare two PCI bus ID strings for same device (bus:device, ignoring function)
  static bool pci_buses_match(char const* a, char const* b) {
    if (!a || !b) return false;
    // Both should be like "0000:XX:YY.Z" — compare up to the function (.Z)
    std::string sa(a), sb(b);
    auto dot_a = sa.rfind('.');
    auto dot_b = sb.rfind('.');
    if (dot_a != std::string::npos) sa = sa.substr(0, dot_a);
    if (dot_b != std::string::npos) sb = sb.substr(0, dot_b);
    return sa == sb;
  }

  /// Detect interconnect type between two GPUs.
  /// Uses /sys/class/kfd/topology/ to check for XGMI links.
  static std::string detect_interconnect_type(int gpu_a, int gpu_b) {
    // On MI300/MI250 with XGMI, the KFD topology exposes link type.
    // Check /sys/class/kfd/topology/nodes/N/io_links/M/type
    // Type 1 = PCIe, Type 2 = XGMI
    //
    // Simple heuristic: if hipDeviceCanAccessPeer is bidirectional AND
    // both GPUs are on the same NUMA node, it's likely XGMI.
    int a_to_b = 0, b_to_a = 0;
    hipDeviceCanAccessPeer(&a_to_b, gpu_a, gpu_b);
    hipDeviceCanAccessPeer(&b_to_a, gpu_b, gpu_a);

    int32_t numa_a = read_sysfs_numa_node(gpu_a);
    int32_t numa_b = read_sysfs_numa_node(gpu_b);

    if (a_to_b && b_to_a && numa_a >= 0 && numa_a == numa_b) {
      // Bidirectional peer access + same NUMA = likely XGMI
      return "XGMI";
    }
    return "PCIe";
  }
};

}  // namespace cucascade::memory
