cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(rocm-cuda-compat VERSION 0.1.0 LANGUAGES CXX HIP)

# ---- cuda-compat-shims: header-only library ----
# Verified on real gfx942/ROCm 7.2.1 (0 compile errors).
add_library(cuda-compat-shims INTERFACE)
target_include_directories(cuda-compat-shims INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cuda-compat-shims/include>)
target_compile_features(cuda-compat-shims INTERFACE cxx_std_20)

# ---- cuco-rocm: header-only library ----
# cuCollections port for ROCm/HIP (bloom_filter, static_set, xxhash).
add_library(cuco-rocm INTERFACE)
target_include_directories(cuco-rocm INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cuco-rocm/include>)
target_compile_features(cuco-rocm INTERFACE cxx_std_20)
target_link_libraries(cuco-rocm INTERFACE cuda-compat-shims)

# ---- cucascade-rocm: header-only stub library ----
# cuCascade compile-only stubs (throw at runtime → CPU fallback).
# Real implementation is WIP.
add_library(cucascade-rocm INTERFACE)
target_include_directories(cucascade-rocm INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cucascade-rocm/include>)
target_compile_features(cucascade-rocm INTERFACE cxx_std_20)
target_link_libraries(cucascade-rocm INTERFACE cuda-compat-shims)

# cucascade-rocm headers include <rmm/...> and <cudf/...>; make those
# dependencies visible to downstream consumers via the exported config.
find_package(rmm CONFIG QUIET)
find_package(cudf CONFIG QUIET)
if(TARGET rmm::rmm)
  target_link_libraries(cucascade-rocm INTERFACE rmm::rmm)
endif()
if(TARGET cudf::cudf)
  target_link_libraries(cucascade-rocm INTERFACE cudf::cudf)
endif()

# ---- Tests ----
option(BUILD_TESTS "Build cuco-rocm tests" OFF)
if(BUILD_TESTS)
  enable_testing()

  # .cu files must be treated as HIP sources
  list(APPEND CMAKE_HIP_SOURCE_FILE_EXTENSIONS cu)

  add_executable(test_bloom_filter cuco-rocm/tests/test_bloom_filter.cu)
  set_source_files_properties(cuco-rocm/tests/test_bloom_filter.cu PROPERTIES LANGUAGE HIP)
  target_link_libraries(test_bloom_filter PRIVATE cuco-rocm)
  set_target_properties(test_bloom_filter PROPERTIES
    HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
  add_test(NAME bloom_filter COMMAND test_bloom_filter)

  add_executable(test_static_set cuco-rocm/tests/test_static_set.cu)
  set_source_files_properties(cuco-rocm/tests/test_static_set.cu PROPERTIES LANGUAGE HIP)
  target_link_libraries(test_static_set PRIVATE cuco-rocm)
  set_target_properties(test_static_set PROPERTIES
    HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
  add_test(NAME static_set COMMAND test_static_set)

  # cucascade-rocm compile + logic test (host-only; no GPU required).
  # Instantiates the real classes to catch API drift from the cuCascade
  # base classes Sirius subclasses.
  add_executable(test_cucascade_compile cucascade-rocm/tests/test_cucascade_compile.cpp)
  target_link_libraries(test_cucascade_compile PRIVATE cucascade-rocm)
  set_target_properties(test_cucascade_compile PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  add_test(NAME cucascade_compile COMMAND test_cucascade_compile)
endif()

# ---- Install ----
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS cuda-compat-shims cuco-rocm cucascade-rocm
        EXPORT rocm-cuda-compat-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY cuda-compat-shims/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY cuco-rocm/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY cucascade-rocm/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT rocm-cuda-compat-targets
        FILE rocm-cuda-compat-targets.cmake
        NAMESPACE rocm-cuda-compat::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocm-cuda-compat)

# Install the config file so find_package(rocm-cuda-compat CONFIG) works
install(FILES cmake/rocm-cuda-compat-config.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocm-cuda-compat)

# Generate + install a version file so find_package(rocm-cuda-compat 0.1 CONFIG)
# version checks work (without it, only unversioned find_package succeeds).
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/rocm-cuda-compat-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rocm-cuda-compat-config-version.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocm-cuda-compat)
