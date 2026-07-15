# rocm-cuda-compat-config.cmake — find_package(rocm-cuda-compat CONFIG REQUIRED)
#
# This file allows downstream projects to consume cuda2rocm via:
#   find_package(rocm-cuda-compat CONFIG REQUIRED)
#   target_link_libraries(my-target PRIVATE rocm-cuda-compat::cuda-compat-shims)

include(CMakeFindDependencyMacro)

# cucascade-rocm headers include <rmm/...> and <cudf/...>. If those targets
# were linked at build time, ensure they are found for consumers too.
find_dependency(rmm CONFIG)
find_dependency(cudf CONFIG)

include("${CMAKE_CURRENT_LIST_DIR}/rocm-cuda-compat-targets.cmake")

# Provide alias targets for convenience
if(TARGET rocm-cuda-compat::cuda-compat-shims)
  message(STATUS "Found rocm-cuda-compat (cuda-compat-shims)")
endif()
if(TARGET rocm-cuda-compat::cuco-rocm)
  message(STATUS "Found rocm-cuda-compat (cuco-rocm)")
endif()
if(TARGET rocm-cuda-compat::cucascade-rocm)
  message(STATUS "Found rocm-cuda-compat (cucascade-rocm)")
endif()
