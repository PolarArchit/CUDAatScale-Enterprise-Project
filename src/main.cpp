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

#include <iostream>
#include <stdexcept>

#include "cli_parser.h"
#include "cuda_common.h"
#include "pipeline_manager.h"

int main(int argc, char* argv[]) {
  try {
    cuda_scale::PipelineOptions options;
    if (!cuda_scale::ParseCommandLine(argc, argv, &options)) {
      cuda_scale::PrintUsage(argv[0]);
      return 1;
    }

    if (options.show_help) {
      cuda_scale::PrintUsage(argv[0]);
      return 0;
    }

    std::cout << "\n======================================================================\n"
              << "     CUDA AT SCALE FOR THE ENTERPRISE - INDEPENDENT COURSE PROJECT     \n"
              << "       Enterprise-Grade Asynchronous Image Processing Pipeline        \n"
              << "======================================================================\n\n";

    // Query and display target CUDA device capabilities.
    cuda_scale::PrintGpuDeviceInfo(options.device_id);

    cuda_scale::PipelineManager manager(options);
    if (!manager.Initialize()) {
      std::cerr << "Fatal: Failed to initialize CUDA pipeline." << std::endl;
      return 1;
    }

    if (options.run_benchmark) {
      if (!manager.RunBenchmarkSuite()) {
        std::cerr << "Fatal: Benchmark execution failed." << std::endl;
        return 1;
      }
    } else {
      if (!manager.RunBatch()) {
        std::cerr << "Fatal: Batch execution failed." << std::endl;
        return 1;
      }
    }

    std::cout << "\n[Success] Enterprise CUDA processing completed successfully.\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "\n[Fatal Exception caught in main]: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "\n[Fatal Unknown Exception caught in main]\n";
    return 1;
  }
}
