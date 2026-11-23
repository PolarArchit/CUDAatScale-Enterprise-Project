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

#ifndef CUDA_AT_SCALE_INCLUDE_PIPELINE_MANAGER_H_
#define CUDA_AT_SCALE_INCLUDE_PIPELINE_MANAGER_H_

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

#include "benchmark.h"
#include "cli_parser.h"
#include "image.h"

namespace cuda_scale {

// State and resources managed per concurrent CUDA stream.
struct StreamResource {
  cudaStream_t stream = nullptr;
  cudaEvent_t h2d_start = nullptr;
  cudaEvent_t h2d_end = nullptr;
  cudaEvent_t kernel_start = nullptr;
  cudaEvent_t kernel_end = nullptr;
  cudaEvent_t d2h_start = nullptr;
  cudaEvent_t d2h_end = nullptr;

  // Device pitched memory pointers
  uint8_t* d_input = nullptr;
  size_t input_pitch = 0;
  uint8_t* d_output = nullptr;
  size_t output_pitch = 0;
  uint8_t* d_intermediate = nullptr;
  size_t intermediate_pitch = 0;

  // Track allocated capacity
  int allocated_width = 0;
  int allocated_height = 0;

  // Pinned host memory buffers for fast DMA transfer
  uint8_t* pinned_host_in = nullptr;
  size_t pinned_in_capacity = 0;
  uint8_t* pinned_host_out = nullptr;
  size_t pinned_out_capacity = 0;
};

// Manages the end-to-end multi-stream batch processing pipeline.
class PipelineManager {
 public:
  explicit PipelineManager(const PipelineOptions& options);
  ~PipelineManager();

  // Non-copyable, non-movable
  PipelineManager(const PipelineManager&) = delete;
  PipelineManager& operator=(const PipelineManager&) = delete;

  // Initializes CUDA device and allocates stream resources.
  bool Initialize();

  // Executes the batch image processing pipeline across all input images.
  bool RunBatch();

  // Executes a comprehensive benchmark comparing sequential CPU vs single-stream vs multi-stream GPU.
  bool RunBenchmarkSuite();

  // Access the benchmark reporter.
  const BenchmarkReporter& Reporter() const { return reporter_; }

 private:
  void CleanupResources();
  void EnsureDeviceCapacity(StreamResource* res, int width, int height);
  void EnsurePinnedCapacity(StreamResource* res, size_t required_bytes);

  PipelineOptions options_;
  std::vector<StreamResource> stream_pool_;
  BenchmarkReporter reporter_;
  bool initialized_ = false;
};

}  // namespace cuda_scale

#endif  // CUDA_AT_SCALE_INCLUDE_PIPELINE_MANAGER_H_
