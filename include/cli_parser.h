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

#ifndef CUDA_AT_SCALE_INCLUDE_CLI_PARSER_H_
#define CUDA_AT_SCALE_INCLUDE_CLI_PARSER_H_

#include <string>

namespace cuda_scale {

// Processing operation mode.
enum class OperationMode {
  kRotate,  // NPP Geometric Rotation (nppiRotate)
  kBlur,    // NPP Box/Gaussian Spatial Filter (nppiFilterBox)
  kSobel,   // Custom CUDA Shared-Memory Sobel 2D Edge Detection
  kAll      // Multi-stage complete pipeline (Blur -> Rotate -> Sobel)
};

// Configuration options parsed from command-line arguments.
struct PipelineOptions {
  std::string input_dir = "data/input";
  std::string output_dir = "output";
  OperationMode operation = OperationMode::kAll;
  double rotation_angle = 45.0;
  int num_streams = 4;
  int batch_size = 0;  // 0 means process all available images.
  bool run_benchmark = false;
  bool verify_with_cpu = false;
  int device_id = 0;
  bool show_help = false;
};

// Parses command-line arguments into PipelineOptions.
bool ParseCommandLine(int argc, char* argv[], PipelineOptions* options);

// Prints usage instructions and parameter documentation.
void PrintUsage(const char* program_name);

// Converts OperationMode enum to descriptive string.
std::string OperationModeToString(OperationMode mode);

}  // namespace cuda_scale

#endif  // CUDA_AT_SCALE_INCLUDE_CLI_PARSER_H_
