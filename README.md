# Enterprise-Scale CUDA NPP Image Processing Pipeline

[![CUDA](https://img.shields.io/badge/CUDA-12.8%20%7C%2011.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B)](https://isocpp.org/)
[![Style](https://img.shields.io/badge/Code%20Style-Google%20C%2B%2B-brightgreen)](https://google.github.io/styleguide/cppguide.html)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **Coursera Capstone Project**: *CUDA at Scale for the Enterprise*  
> An enterprise-grade batch image processing pipeline that takes the course project template (`imageRotationNPP`) and scales it to process batches of **100+ images concurrently** using **NVIDIA Performance Primitives (NPP)** and asynchronous **CUDA Streams**.

---

## 1. Project Overview

In high-throughput enterprise environments (such as automated manufacturing inspection, medical imaging, and remote sensing), processing images one by one causes severe bottlenecks because the GPU sits idle while data transfers across the PCIe bus.

This project scales NVIDIA's official **NPP (NVIDIA Performance Primitives)** image processing library to process an entire dataset of **125 images** (over 38 MB) in a single execution. By pipelining data transfers and computation across multiple asynchronous CUDA streams (`cudaStream_t`) using pinned host memory, the pipeline reaches over **390 images/second** throughput on an NVIDIA RTX GPU.

---

## 2. Key Features

- **Built on Course Template**: Uses NVIDIA NPP primitives (`nppiRotate` and `nppiFilterBox`) as demonstrated in the official course project template.
- **Enterprise Batch Scaling**: Processes **125 input images** (256x256, 512x512, and 1024x1024 resolutions) in a single software execution.
- **Asynchronous Multi-Stream Concurrency**: Overlaps Host-to-Device (H2D) transfer, NPP execution, and Device-to-Host (D2H) transfer across 4 to 8 concurrent CUDA streams.
- **Pitched 2D Device Memory**: Uses `cudaMallocPitch` and `cudaMemcpy2DAsync` to guarantee hardware row alignment.
- **Self-Contained Image I/O**: Portable binary Netpbm (PGM/PPM) reader and writer compiling without third-party dependencies.
- **CLI with Arguments**: Full command-line interface supporting `--input_dir`, `--output_dir`, `--operation`, `--angle`, `--streams`, and `--benchmark`.
- **Google C++ Style Guide Compliant**: Formatted cleanly with `.clang-format`.

---

## 3. Repository Structure

```
CudaAtScale/
├── .clang-format                  # Google C++ Style Guide format definition
├── .gitignore                     # Git configuration ignoring build binaries
├── CMakeLists.txt                 # Cross-platform CMake build configuration
├── Makefile                       # Production Makefile for Linux / Coursera Lab
├── build.bat                      # Windows NVCC compilation script
├── run.sh                         # Linux execution script with automated test runner
├── run.bat                        # Windows execution script
├── README.md                      # Project documentation
├── SUBMISSION.md                  # Coursera peer review submission text
├── include/                       # Header files
│   ├── benchmark.h                # Timer and CSV reporting
│   ├── cli_parser.h               # CLI argument parser
│   ├── cuda_common.h              # CUDA_CHECK and NPP_CHECK error handling macros
│   ├── image.h                    # Image data structure
│   ├── image_io.h                 # Binary Netpbm (PGM/PPM) parser & serializer
│   ├── npp_pipeline.h             # NVIDIA Performance Primitives wrapper
│   └── pipeline_manager.h         # Multi-stream batch pipeline orchestrator
├── src/                           # Implementation source files
│   ├── cli_parser.cpp             # CLI argument parser implementation
│   ├── image_io.cpp               # Binary Netpbm reader/writer
│   ├── main.cpp                   # Main entry point and CLI orchestration
│   ├── npp_pipeline.cpp           # NPP rotation and filter implementation
│   └── pipeline_manager.cpp       # Asynchronous multi-stream manager
├── tools/                         # Utilities
│   ├── generate_dataset.py        # Procedural generator creating 125 test images
│   └── generate_visual_artifacts.py # Generates side-by-side comparison images
├── data/
│   └── input/                     # 125 input test images (256x256, 512x512, 1024x1024)
├── output/                        # 125 processed output images & benchmark_results.csv
└── artifacts/                     # Proof of execution artifacts for peer grading
    ├── execution_proof.log        # Complete execution log trace of 125 images
    ├── comparison_sample_concentric_512.pgm
    ├── comparison_sample_checkerboard_512.pgm
    └── comparison_sample_highres_1024.pgm
```

---

## 4. Build & Run Instructions

### Linux / Coursera Lab Environment
```bash
# 1. Compile with Makefile
make -j$(nproc)

# 2. Run batch NPP image rotation on all 125 images
bash run.sh

# 3. Or pass custom arguments
./bin/cuda_enterprise_pipeline --input_dir data/input --output_dir output --operation rotate --angle 45.0 --streams 4
```

### Windows Environment
```cmd
build.bat
run.bat
```

---

## 5. Command-Line Options

```
Options:
  --input_dir <path>       Directory containing input images (default: data/input)
  --output_dir <path>      Directory for processed output images (default: output)
  --operation <mode>       NPP operation to execute:
                             rotate : NVIDIA NPP geometric image rotation (default)
                             filter : NVIDIA NPP spatial box filter
                             all    : Full NPP pipeline (filter + rotate)
  --angle <degrees>        Rotation angle in degrees for NPP rotate (default: 45.0)
  --streams <count>        Number of concurrent CUDA streams (default: 4)
  --batch_size <N>         Max images to process in run (0 = all images) (default: 0)
  --benchmark              Execute 1-stream vs 4-stream vs 8-stream scaling benchmark
  --device <id>            Target CUDA GPU device ID (default: 0)
  --help, -h               Display this usage help message and exit
```

---

## 6. Performance Benchmarks

Executed on an **NVIDIA GeForce RTX 4060 GPU** across **125 images** (38.12 MB):

| Configuration | Concurrency Mode | Total Time | Throughput | Bandwidth | Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1 Stream** | Sequential GPU | 365.25 ms | 342.2 img/s | 104.4 MB/s | 1.00x |
| **4 Streams** | Concurrent Pipelined | 338.16 ms | 369.6 img/s | 112.7 MB/s | 1.08x |
| **8 Streams** | Concurrent Pipelined | **313.32 ms** | **399.0 img/s** | **121.7 MB/s** | **1.16x** |

---

## 7. Proof of Execution Artifacts

- **`artifacts/execution_proof.log`**: Full console execution log showing per-image timing across all 125 files.
- **`output/benchmark_results.csv`**: CSV spreadsheet with per-file latency, image dimensions, and throughput.
- **`output/processed_*.pgm` & `output/processed_*.ppm`**: All 125 generated output images.
- **`artifacts/comparison_sample_*.png` & `.pgm`**: Side-by-side visual comparison artifacts (original vs rotated).

---

## 8. Engineering Challenges, Debugging & Lessons Learned

### A. The Rotation "Corner-Shift" Defect & Coordinate Centering
* **Defect Identified:** In early test runs of $45^\circ$ rotation using `nppiRotate`, visual output inspection revealed that the rotated image was displaced into the top-right corner, leaving most of the canvas blank black space. Because image coordinates place $(0,0)$ at the top-left with positive $Y$ pointing downwards, applying standard textbook rotation formulas around $(w/2, h/2)$ produced an incorrect translation offset.
* **Resolution:** Visual verification of the generated artifact (`comparison_sample_checkerboard_512.png`) caught the error. I resolved it by integrating NVIDIA's utility primitive `nppiGetRotateBound()`, which computes the exact axis-aligned bounding box of the rotated quadrilateral. Calculating centering shifts based on this bounding box centered the rotated diamond lattice into a clean, symmetrical octagon inside the canvas.

### B. Hardware Memory Pitch Alignment (`NPP_STEP_ERROR`)
* **Problem:** Allocating unpitched 1D memory buffers caused NPP to fail with `NPP_STEP_ERROR` due to row stride misalignments.
* **Resolution:** Switched to pitched 2D device allocations via `cudaMallocPitch` and `cudaMemcpy2DAsync`, guaranteeing 128-byte alignment and optimal memory coalescing.

### C. Overcoming DMA Serialization with Pinned Memory
* **Problem:** Standard pageable host buffers forced the CUDA driver to perform staged synchronous copies, preventing streams from running concurrently.
* **Resolution:** Staged images in page-locked pinned host memory (`cudaHostAlloc`), enabling the GPU's DMA copy engines to transfer data concurrently over PCIe while compute cores execute.
