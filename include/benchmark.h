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

#ifndef CUDA_AT_SCALE_INCLUDE_BENCHMARK_H_
#define CUDA_AT_SCALE_INCLUDE_BENCHMARK_H_

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace cuda_scale {

// High-resolution CPU timer.
class Timer {
 public:
  Timer() { Reset(); }

  void Reset() {
    start_time_ = std::chrono::high_resolution_clock::now();
  }

  double ElapsedMilliseconds() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(now - start_time_).count();
  }

  double ElapsedMicroseconds() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(now - start_time_).count();
  }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

// Records metrics for an individual image execution.
struct ImageBenchmarkRecord {
  std::string filename;
  int width = 0;
  int height = 0;
  int channels = 1;
  double h2d_ms = 0.0;
  double kernel_ms = 0.0;
  double d2h_ms = 0.0;
  double total_ms = 0.0;
  double cpu_ref_ms = 0.0;
  double psnr_db = 0.0;
};

// Aggregates batch benchmark statistics and exports reports.
class BenchmarkReporter {
 public:
  void AddRecord(const ImageBenchmarkRecord& rec) {
    records_.push_back(rec);
  }

  const std::vector<ImageBenchmarkRecord>& Records() const {
    return records_;
  }

  size_t Count() const {
    return records_.size();
  }

  // Export records to CSV file for audit and peer evaluation.
  bool ExportToCSV(const std::string& csv_filepath) const {
    std::ofstream out(csv_filepath);
    if (!out.is_open()) {
      std::cerr << "[Benchmark] Failed to write CSV: " << csv_filepath << std::endl;
      return false;
    }

    out << "Filename,Width,Height,Channels,H2D_ms,Compute_ms,D2H_ms,Total_ms,"
        << "CPU_Ref_ms,Speedup,PSNR_dB,Throughput_MBs\n";

    for (const auto& r : records_) {
      double bytes = static_cast<double>(r.width * r.height * r.channels);
      double mb = bytes / (1024.0 * 1024.0);
      double throughput = (r.total_ms > 0.0) ? (mb / (r.total_ms / 1000.0)) : 0.0;
      double speedup = (r.total_ms > 0.0 && r.cpu_ref_ms > 0.0)
                           ? (r.cpu_ref_ms / r.total_ms)
                           : 0.0;

      out << r.filename << ","
          << r.width << ","
          << r.height << ","
          << r.channels << ","
          << std::fixed << std::setprecision(4)
          << r.h2d_ms << ","
          << r.kernel_ms << ","
          << r.d2h_ms << ","
          << r.total_ms << ","
          << r.cpu_ref_ms << ","
          << std::setprecision(2)
          << speedup << ","
          << r.psnr_db << ","
          << throughput << "\n";
    }

    return true;
  }

  // Prints formatted batch performance summary to stdout.
  void PrintSummary(double wall_clock_total_ms, int num_streams) const {
    if (records_.empty()) {
      std::cout << "[Benchmark] No records logged.\n";
      return;
    }

    double total_bytes = 0.0;
    double sum_gpu_ms = 0.0;
    double sum_cpu_ms = 0.0;
    double sum_kernel_ms = 0.0;
    double sum_h2d_ms = 0.0;
    double sum_d2h_ms = 0.0;

    for (const auto& r : records_) {
      total_bytes += static_cast<double>(r.width * r.height * r.channels);
      sum_gpu_ms += r.total_ms;
      sum_cpu_ms += r.cpu_ref_ms;
      sum_kernel_ms += r.kernel_ms;
      sum_h2d_ms += r.h2d_ms;
      sum_d2h_ms += r.d2h_ms;
    }

    const double total_mb = total_bytes / (1024.0 * 1024.0);
    const double wall_clock_sec = wall_clock_total_ms / 1000.0;
    const double effective_fps = (wall_clock_sec > 0.0) ? (records_.size() / wall_clock_sec) : 0.0;
    const double effective_throughput_mb_s = (wall_clock_sec > 0.0) ? (total_mb / wall_clock_sec) : 0.0;
    const double avg_gpu_ms = sum_gpu_ms / records_.size();
    const double avg_kernel_ms = sum_kernel_ms / records_.size();

    std::cout << "\n"
              << "======================================================================\n"
              << "          ENTERPRISE BATCH PROCESSING PERFORMANCE SUMMARY             \n"
              << "======================================================================\n"
              << " Total Images Processed       : " << records_.size() << " images\n"
              << " Total Image Data Volume      : " << std::fixed << std::setprecision(2)
              << total_mb << " MB\n"
              << " Active CUDA Streams          : " << num_streams << " concurrent streams\n"
              << " Total Pipeline Wall Time     : " << std::setprecision(2)
              << wall_clock_total_ms << " ms (" << wall_clock_sec << " s)\n"
              << " Effective Processing Rate    : " << std::setprecision(1)
              << effective_fps << " images/sec\n"
              << " Aggregate Memory Throughput  : " << std::setprecision(2)
              << effective_throughput_mb_s << " MB/sec\n"
              << "----------------------------------------------------------------------\n"
              << " Per-Image Latency Breakdown (Mean):\n"
              << "   - Host-to-Device Copy (H2D): " << std::setprecision(3)
              << (sum_h2d_ms / records_.size()) << " ms\n"
              << "   - GPU Compute / Kernel     : " << (sum_kernel_ms / records_.size()) << " ms\n"
              << "   - Device-to-Host Copy (D2H): " << (sum_d2h_ms / records_.size()) << " ms\n"
              << "   - End-to-End Latency (Avg) : " << avg_gpu_ms << " ms\n";

    if (sum_cpu_ms > 0.0) {
      const double avg_cpu_ms = sum_cpu_ms / records_.size();
      const double overall_speedup = sum_cpu_ms / wall_clock_total_ms;
      std::cout << "----------------------------------------------------------------------\n"
                << " CPU Baseline Comparison:\n"
                << "   - Sequential CPU Mean Time : " << avg_cpu_ms << " ms/image\n"
                << "   - GPU Pipeline Speedup     : " << std::setprecision(2)
                << overall_speedup << "x faster than CPU\n";
    }

    std::cout << "======================================================================\n"
              << std::endl;
  }

 private:
  std::vector<ImageBenchmarkRecord> records_;
};

}  // namespace cuda_scale

#endif  // CUDA_AT_SCALE_INCLUDE_BENCHMARK_H_
