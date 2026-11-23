// Copyright 2026 Enterprise CUDA Team. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CUDA_AT_SCALE_INCLUDE_CUDA_COMMON_H_
#define CUDA_AT_SCALE_INCLUDE_CUDA_COMMON_H_

#include <cuda_runtime.h>
#include <npp.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cuda_scale {

// Macro for checking CUDA runtime API calls.
#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    const cudaError_t error = (call);                                    \
    if (error != cudaSuccess) {                                          \
      std::ostringstream ss;                                             \
      ss << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - "    \
         << cudaGetErrorString(error) << " (" << error << ")";           \
      std::cerr << ss.str() << std::endl;                                \
      throw std::runtime_error(ss.str());                                \
    }                                                                    \
  } while (0)

// Helper function to convert NPP status code to human-readable string.
inline const char* NppGetErrorString(NppStatus status) {
  switch (status) {
    case NPP_SUCCESS:
      return "NPP_SUCCESS: Operation completed without errors";
    case NPP_NOT_SUPPORTED_MODE_ERROR:
      return "NPP_NOT_SUPPORTED_MODE_ERROR: Operation not supported";
    case NPP_INVALID_HOST_POINTER_ERROR:
      return "NPP_INVALID_HOST_POINTER_ERROR: Null host pointer";
    case NPP_INVALID_DEVICE_POINTER_ERROR:
      return "NPP_INVALID_DEVICE_POINTER_ERROR: Null device pointer";
    case NPP_MEMCPY_ERROR:
      return "NPP_MEMCPY_ERROR: Memory copy error";
    case NPP_STEP_ERROR:
      return "NPP_STEP_ERROR: Step is less than or equal to zero";
    case NPP_SIZE_ERROR:
      return "NPP_SIZE_ERROR: Size is less than or equal to zero";
    case NPP_NULL_POINTER_ERROR:
      return "NPP_NULL_POINTER_ERROR: Null pointer passed";
    case NPP_CUDA_KERNEL_EXECUTION_ERROR:
      return "NPP_CUDA_KERNEL_EXECUTION_ERROR: Kernel execution failed";
    case NPP_NOT_IMPLEMENTED_ERROR:
      return "NPP_NOT_IMPLEMENTED_ERROR: Feature not implemented";
    case NPP_ERROR:
      return "NPP_ERROR: Unknown NPP error";
    case NPP_WRONG_INTERSECTION_QUAD_WARNING:
      return "NPP_WRONG_INTERSECTION_QUAD_WARNING: Quad intersection warning";
    case NPP_AFFINE_QUAD_INCORRECT_WARNING:
      return "NPP_AFFINE_QUAD_INCORRECT_WARNING: Quad incorrect warning";
    default:
      return "NPP_STATUS: Status code returned";
  }
}

// Macro for checking NVIDIA Performance Primitives (NPP) calls.
#define NPP_CHECK(call)                                                  \
  do {                                                                   \
    const NppStatus status = (call);                                     \
    if (status != NPP_SUCCESS) {                                         \
      std::ostringstream ss;                                             \
      ss << "NPP Error at " << __FILE__ << ":" << __LINE__ << " - "     \
         << NppGetErrorString(status) << " (" << status << ")";          \
      std::cerr << ss.str() << std::endl;                                \
      throw std::runtime_error(ss.str());                                \
    }                                                                    \
  } while (0)

// Prints GPU device specifications and capabilities.
inline void PrintGpuDeviceInfo(int device_id = 0) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

  std::cout << "========================================================\n"
            << "              CUDA GPU DEVICE SPECIFICATIONS            \n"
            << "========================================================\n"
            << " Device ID               : " << device_id << "\n"
            << " Device Name             : " << prop.name << "\n"
            << " Compute Capability      : " << prop.major << "." << prop.minor << "\n"
            << " Total Global Memory     : " << (prop.totalGlobalMem / (1024 * 1024))
            << " MB\n"
            << " Multiprocessors (SMs)   : " << prop.multiProcessorCount << "\n"
            << " Max Threads Per Block   : " << prop.maxThreadsPerBlock << "\n"
            << " Max Threads Per SM      : " << prop.maxThreadsPerMultiProcessor << "\n"
            << " Warp Size               : " << prop.warpSize << "\n"
            << " Memory Clock Rate       : " << (prop.memoryClockRate * 1e-3f) << " MHz\n"
            << " Memory Bus Width        : " << prop.memoryBusWidth << " bits\n"
            << " Async Engine Count      : " << prop.asyncEngineCount << "\n"
            << " Overlap Engine (Streams): " << (prop.asyncEngineCount > 1 ? "Yes (Full Duplex)" : "Yes") << "\n"
            << "========================================================\n"
            << std::endl;
}

}  // namespace cuda_scale

#endif  // CUDA_AT_SCALE_INCLUDE_CUDA_COMMON_H_
