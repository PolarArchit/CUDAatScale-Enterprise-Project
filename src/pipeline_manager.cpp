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

#include "pipeline_manager.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark.h"
#include "cli_parser.h"
#include "cuda_common.h"
#include "image.h"
#include "image_io.h"
#include "npp_pipeline.h"

namespace cuda_scale {

namespace {

constexpr int kInitialWidth = 1024;
constexpr int kInitialHeight = 1024;

}  // namespace

PipelineManager::PipelineManager(const PipelineOptions& options)
    : options_(options) {}

PipelineManager::~PipelineManager() {
  CleanupResources();
}

bool PipelineManager::Initialize() {
  if (initialized_) return true;

  CUDA_CHECK(cudaSetDevice(options_.device_id));

  stream_pool_.resize(options_.num_streams);
  for (int i = 0; i < options_.num_streams; ++i) {
    auto& res = stream_pool_[i];
    CUDA_CHECK(cudaStreamCreateWithFlags(&res.stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaEventCreate(&res.h2d_start));
    CUDA_CHECK(cudaEventCreate(&res.h2d_end));
    CUDA_CHECK(cudaEventCreate(&res.kernel_start));
    CUDA_CHECK(cudaEventCreate(&res.kernel_end));
    CUDA_CHECK(cudaEventCreate(&res.d2h_start));
    CUDA_CHECK(cudaEventCreate(&res.d2h_end));

    EnsureDeviceCapacity(&res, kInitialWidth, kInitialHeight);
    EnsurePinnedCapacity(&res, static_cast<size_t>(kInitialWidth) * kInitialHeight * 3);
  }

  initialized_ = true;
  return true;
}

void PipelineManager::EnsureDeviceCapacity(StreamResource* res, int width, int height) {
  if (!res) return;

  if (res->allocated_width >= width && res->allocated_height >= height) {
    return;
  }

  int new_w = std::max(width, res->allocated_width);
  int new_h = std::max(height, res->allocated_height);

  if (res->d_input) CUDA_CHECK(cudaFree(res->d_input));
  if (res->d_output) CUDA_CHECK(cudaFree(res->d_output));
  if (res->d_intermediate) CUDA_CHECK(cudaFree(res->d_intermediate));

  CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&res->d_input),
                             &res->input_pitch,
                             new_w * 3, new_h));
  CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&res->d_output),
                             &res->output_pitch,
                             new_w * 3, new_h));
  CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&res->d_intermediate),
                             &res->intermediate_pitch,
                             new_w * 3, new_h));

  res->allocated_width = new_w;
  res->allocated_height = new_h;
}

void PipelineManager::EnsurePinnedCapacity(StreamResource* res, size_t required_bytes) {
  if (!res) return;

  if (res->pinned_in_capacity < required_bytes) {
    if (res->pinned_host_in) CUDA_CHECK(cudaFreeHost(res->pinned_host_in));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&res->pinned_host_in),
                             required_bytes, cudaHostAllocDefault));
    res->pinned_in_capacity = required_bytes;
  }

  if (res->pinned_out_capacity < required_bytes) {
    if (res->pinned_host_out) CUDA_CHECK(cudaFreeHost(res->pinned_host_out));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&res->pinned_host_out),
                             required_bytes, cudaHostAllocDefault));
    res->pinned_out_capacity = required_bytes;
  }
}

void PipelineManager::CleanupResources() {
  for (auto& res : stream_pool_) {
    if (res.stream) {
      cudaStreamSynchronize(res.stream);
      cudaStreamDestroy(res.stream);
      res.stream = nullptr;
    }
    if (res.h2d_start) cudaEventDestroy(res.h2d_start);
    if (res.h2d_end) cudaEventDestroy(res.h2d_end);
    if (res.kernel_start) cudaEventDestroy(res.kernel_start);
    if (res.kernel_end) cudaEventDestroy(res.kernel_end);
    if (res.d2h_start) cudaEventDestroy(res.d2h_start);
    if (res.d2h_end) cudaEventDestroy(res.d2h_end);

    if (res.d_input) cudaFree(res.d_input);
    if (res.d_output) cudaFree(res.d_output);
    if (res.d_intermediate) cudaFree(res.d_intermediate);

    if (res.pinned_host_in) cudaFreeHost(res.pinned_host_in);
    if (res.pinned_host_out) cudaFreeHost(res.pinned_host_out);
  }
  stream_pool_.clear();
  initialized_ = false;
}

bool PipelineManager::RunBatch() {
  if (!initialized_ && !Initialize()) {
    std::cerr << "[Pipeline] Failed to initialize CUDA pipeline." << std::endl;
    return false;
  }

  std::vector<std::string> file_paths = GetImageFilesInDirectory(options_.input_dir);
  if (file_paths.empty()) {
    std::cerr << "[Pipeline] No image files found in input directory: "
              << options_.input_dir << std::endl;
    return false;
  }

  if (options_.batch_size > 0 &&
      options_.batch_size < static_cast<int>(file_paths.size())) {
    file_paths.resize(options_.batch_size);
  }

  std::filesystem::create_directories(options_.output_dir);

  std::cout << "\n======================================================================\n"
            << "  Starting Batch Image Execution: " << file_paths.size()
            << " images | Operation: " << OperationModeToString(options_.operation) << "\n"
            << "  Concurrent Streams: " << options_.num_streams << " | Target Device: "
            << options_.device_id << "\n"
            << "======================================================================\n";

  struct InFlightTask {
    std::string filepath;
    std::string out_filepath;
    Image input_image;
    Image output_image;
    bool active = false;
  };

  std::vector<InFlightTask> in_flight(options_.num_streams);

  auto complete_task = [&](int s_idx) {
    auto& task = in_flight[s_idx];
    if (!task.active) return;

    auto& res = stream_pool_[s_idx];
    CUDA_CHECK(cudaStreamSynchronize(res.stream));

    float h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, res.h2d_start, res.h2d_end));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, res.kernel_start, res.kernel_end));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, res.d2h_start, res.d2h_end));

    std::memcpy(task.output_image.data.data(), res.pinned_host_out,
                task.output_image.SizeInBytes());

    SaveImageFile(task.out_filepath, task.output_image);

    ImageBenchmarkRecord rec;
    rec.filename = std::filesystem::path(task.filepath).filename().string();
    rec.width = task.input_image.width;
    rec.height = task.input_image.height;
    rec.channels = task.input_image.channels;
    rec.h2d_ms = h2d_ms;
    rec.compute_ms = kernel_ms;
    rec.d2h_ms = d2h_ms;
    rec.total_ms = h2d_ms + kernel_ms + d2h_ms;

    reporter_.AddRecord(rec);

    std::cout << " [Processed NPP] " << std::left << std::setw(22) << rec.filename
              << " | " << rec.width << "x" << rec.height << " (" << rec.channels << "ch)"
              << " | H2D: " << std::fixed << std::setprecision(2) << rec.h2d_ms << "ms"
              << " | NPP: " << rec.compute_ms << "ms"
              << " | D2H: " << rec.d2h_ms << "ms"
              << " | Stream: " << s_idx << "\n";

    task.active = false;
  };

  Timer wall_clock;

  for (size_t i = 0; i < file_paths.size(); ++i) {
    const int stream_idx = static_cast<int>(i % options_.num_streams);
    auto& res = stream_pool_[stream_idx];
    auto& task = in_flight[stream_idx];

    if (task.active) {
      complete_task(stream_idx);
    }

    task.filepath = file_paths[i];
    std::string filename = std::filesystem::path(task.filepath).filename().string();
    task.out_filepath = (std::filesystem::path(options_.output_dir) /
                         ("processed_" + filename)).string();

    if (!LoadImageFile(task.filepath, &task.input_image)) {
      std::cerr << "[Pipeline] Failed to load image: " << task.filepath << std::endl;
      continue;
    }

    const int w = task.input_image.width;
    const int h = task.input_image.height;
    const int c = task.input_image.channels;
    task.output_image.Allocate(w, h, c);

    EnsureDeviceCapacity(&res, w, h);
    EnsurePinnedCapacity(&res, task.input_image.SizeInBytes());

    std::memcpy(res.pinned_host_in, task.input_image.data.data(),
                task.input_image.SizeInBytes());

    // Stage 1: Async H2D Transfer
    CUDA_CHECK(cudaEventRecord(res.h2d_start, res.stream));
    CUDA_CHECK(cudaMemcpy2DAsync(res.d_input, res.input_pitch,
                                 res.pinned_host_in, w * c,
                                 w * c, h,
                                 cudaMemcpyHostToDevice, res.stream));
    CUDA_CHECK(cudaEventRecord(res.h2d_end, res.stream));

    // Stage 2: NVIDIA NPP Library Execution
    CUDA_CHECK(cudaEventRecord(res.kernel_start, res.stream));
    uint8_t* final_output_d = res.d_output;

    switch (options_.operation) {
      case OperationMode::kRotate:
        ExecuteNppRotate(res.d_input, res.input_pitch,
                         res.d_output, res.output_pitch,
                         w, h, c, options_.rotation_angle, res.stream);
        final_output_d = res.d_output;
        break;

      case OperationMode::kFilter:
        ExecuteNppBoxFilter(res.d_input, res.input_pitch,
                            res.d_output, res.output_pitch,
                            w, h, c, 3, res.stream);
        final_output_d = res.d_output;
        break;

      case OperationMode::kAll:
        // Pipelined: Box Filter -> Rotation
        ExecuteNppBoxFilter(res.d_input, res.input_pitch,
                            res.d_intermediate, res.intermediate_pitch,
                            w, h, c, 3, res.stream);
        ExecuteNppRotate(res.d_intermediate, res.intermediate_pitch,
                         res.d_output, res.output_pitch,
                         w, h, c, options_.rotation_angle, res.stream);
        final_output_d = res.d_output;
        break;
    }
    CUDA_CHECK(cudaEventRecord(res.kernel_end, res.stream));

    // Stage 3: Async D2H Transfer
    CUDA_CHECK(cudaEventRecord(res.d2h_start, res.stream));
    CUDA_CHECK(cudaMemcpy2DAsync(res.pinned_host_out, w * c,
                                 final_output_d, res.output_pitch,
                                 w * c, h,
                                 cudaMemcpyDeviceToHost, res.stream));
    CUDA_CHECK(cudaEventRecord(res.d2h_end, res.stream));

    task.active = true;
  }

  for (int s = 0; s < options_.num_streams; ++s) {
    if (in_flight[s].active) {
      complete_task(s);
    }
  }

  double total_wall_ms = wall_clock.ElapsedMilliseconds();

  std::string csv_path = (std::filesystem::path(options_.output_dir) /
                          "benchmark_results.csv").string();
  reporter_.ExportToCSV(csv_path);
  std::cout << "[Pipeline] Detailed CSV benchmark exported to: " << csv_path << "\n";

  reporter_.PrintSummary(total_wall_ms, options_.num_streams);

  return true;
}

bool PipelineManager::RunBenchmarkSuite() {
  std::cout << "\n======================================================================\n"
            << "              RUNNING COMPREHENSIVE BENCHMARK SUITE                   \n"
            << "======================================================================\n";

  PipelineOptions opt_1 = options_;
  opt_1.num_streams = 1;
  PipelineManager mgr_1(opt_1);
  mgr_1.Initialize();
  Timer t1;
  mgr_1.RunBatch();
  double time_1_stream = t1.ElapsedMilliseconds();

  PipelineOptions opt_4 = options_;
  opt_4.num_streams = 4;
  PipelineManager mgr_4(opt_4);
  mgr_4.Initialize();
  Timer t4;
  mgr_4.RunBatch();
  double time_4_stream = t4.ElapsedMilliseconds();

  PipelineOptions opt_8 = options_;
  opt_8.num_streams = 8;
  PipelineManager mgr_8(opt_8);
  mgr_8.Initialize();
  Timer t8;
  mgr_8.RunBatch();
  double time_8_stream = t8.ElapsedMilliseconds();

  std::cout << "\n======================================================================\n"
            << "               CONCURRENCY SCALING BENCHMARK SUMMARY                  \n"
            << "======================================================================\n"
            << " 1 Stream  (Sequential GPU)  : " << std::fixed << std::setprecision(2)
            << time_1_stream << " ms (Baseline: 1.00x)\n"
            << " 4 Streams (Concurrent GPU)  : " << time_4_stream << " ms (Speedup: "
            << (time_1_stream / time_4_stream) << "x)\n"
            << " 8 Streams (Concurrent GPU)  : " << time_8_stream << " ms (Speedup: "
            << (time_1_stream / time_8_stream) << "x)\n"
            << "======================================================================\n\n";

  return true;
}

}  // namespace cuda_scale
