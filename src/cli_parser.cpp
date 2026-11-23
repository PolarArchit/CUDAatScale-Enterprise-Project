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

#include "cli_parser.h"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace cuda_scale {

std::string OperationModeToString(OperationMode mode) {
  switch (mode) {
    case OperationMode::kRotate:
      return "NPP Image Rotation (nppiRotate)";
    case OperationMode::kFilter:
      return "NPP Box Filter (nppiFilterBox)";
    case OperationMode::kAll:
      return "Full NPP Pipeline (nppiFilterBox + nppiRotate)";
  }
  return "Unknown";
}

void PrintUsage(const char* program_name) {
  std::cout
      << "======================================================================\n"
      << "  Enterprise CUDA at Scale - Batched NPP Image Processing CLI         \n"
      << "======================================================================\n"
      << "Usage:\n"
      << "  " << program_name << " [options]\n\n"
      << "Options:\n"
      << "  --input_dir <path>       Directory containing input images (default: data/input)\n"
      << "  --output_dir <path>      Directory for processed output images (default: output)\n"
      << "  --operation <mode>       NPP operation to execute:\n"
      << "                             rotate : NVIDIA NPP geometric image rotation (default)\n"
      << "                             filter : NVIDIA NPP spatial box filter\n"
      << "                             all    : Full NPP pipeline (filter + rotate)\n"
      << "  --angle <degrees>        Rotation angle in degrees for NPP rotate (default: 45.0)\n"
      << "  --streams <count>        Number of concurrent CUDA streams (default: 4)\n"
      << "  --batch_size <N>         Max images to process in run (0 = all images) (default: 0)\n"
      << "  --benchmark              Execute 1-stream vs 4-stream vs 8-stream scaling benchmark\n"
      << "  --device <id>            Target CUDA GPU device ID (default: 0)\n"
      << "  --help, -h               Display this usage help message and exit\n"
      << "======================================================================\n"
      << "Example:\n"
      << "  " << program_name << " --input_dir data/input --output_dir output --operation rotate --angle 45.0\n"
      << "======================================================================\n";
}

bool ParseCommandLine(int argc, char* argv[], PipelineOptions* options) {
  if (!options) return false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      options->show_help = true;
      return true;
    } else if (arg == "--input_dir") {
      if (i + 1 < argc) {
        options->input_dir = argv[++i];
      } else {
        std::cerr << "Error: --input_dir requires a directory path argument.\n";
        return false;
      }
    } else if (arg == "--output_dir") {
      if (i + 1 < argc) {
        options->output_dir = argv[++i];
      } else {
        std::cerr << "Error: --output_dir requires a directory path argument.\n";
        return false;
      }
    } else if (arg == "--operation") {
      if (i + 1 < argc) {
        std::string mode_str = argv[++i];
        std::transform(mode_str.begin(), mode_str.end(), mode_str.begin(), ::tolower);
        if (mode_str == "rotate") {
          options->operation = OperationMode::kRotate;
        } else if (mode_str == "filter" || mode_str == "blur") {
          options->operation = OperationMode::kFilter;
        } else if (mode_str == "all") {
          options->operation = OperationMode::kAll;
        } else {
          std::cerr << "Error: Unknown operation '" << mode_str
                    << "'. Valid options are: rotate, filter, all.\n";
          return false;
        }
      } else {
        std::cerr << "Error: --operation requires an operation name.\n";
        return false;
      }
    } else if (arg == "--angle") {
      if (i + 1 < argc) {
        options->rotation_angle = std::atof(argv[++i]);
      } else {
        std::cerr << "Error: --angle requires a numeric value.\n";
        return false;
      }
    } else if (arg == "--streams") {
      if (i + 1 < argc) {
        options->num_streams = std::atoi(argv[++i]);
        if (options->num_streams < 1) {
          std::cerr << "Error: --streams must be at least 1.\n";
          return false;
        }
      } else {
        std::cerr << "Error: --streams requires an integer count.\n";
        return false;
      }
    } else if (arg == "--batch_size") {
      if (i + 1 < argc) {
        options->batch_size = std::atoi(argv[++i]);
        if (options->batch_size < 0) {
          std::cerr << "Error: --batch_size cannot be negative.\n";
          return false;
        }
      } else {
        std::cerr << "Error: --batch_size requires an integer.\n";
        return false;
      }
    } else if (arg == "--device") {
      if (i + 1 < argc) {
        options->device_id = std::atoi(argv[++i]);
      } else {
        std::cerr << "Error: --device requires a device ID integer.\n";
        return false;
      }
    } else if (arg == "--benchmark") {
      options->run_benchmark = true;
    } else {
      std::cerr << "Warning: Unknown command-line option ignored: " << arg << "\n";
    }
  }

  return true;
}

}  // namespace cuda_scale
