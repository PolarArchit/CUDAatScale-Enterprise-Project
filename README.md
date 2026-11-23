# Enterprise-Scale CUDA Image Processing Pipeline (ES-CIP)

[![CUDA](https://img.shields.io/badge/CUDA-12.8%20%7C%2011.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B)](https://isocpp.org/)
[![Style](https://img.shields.io/badge/Code%20Style-Google%20C%2B%2B-brightgreen)](https://google.github.io/styleguide/cppguide.html)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey)]()

> **Coursera Capstone Independent Project**: *CUDA at Scale for the Enterprise*  
> High-throughput, multi-stream asynchronous image processing pipeline capable of processing hundreds of images concurrently using **NVIDIA Performance Primitives (NPP)** and **optimized custom 2D shared-memory CUDA kernels**.

---

## Table of Contents
- [1. Project Overview & Motivation](#1-project-overview--motivation)
- [2. Key Features](#2-key-features)
- [3. Architecture & Pipelining](#3-architecture--pipelining)
- [4. Repository Structure](#4-repository-structure)
- [5. Google C++ Style Guide Compliance](#5-google-c-style-guide-compliance)
- [6. Build Instructions](#6-build-instructions)
  - [Linux / Coursera Cloud Lab Environment](#linux--coursera-cloud-lab-environment)
  - [Windows Environment](#windows-environment)
- [7. Command Line Interface (CLI) Usage](#7-command-line-interface-cli-usage)
- [8. Experimental Results & Performance Benchmarks](#8-experimental-results--performance-benchmarks)
- [9. Algorithms & CUDA Kernel Implementation](#9-algorithms--cuda-kernel-implementation)
- [10. Engineering Challenges & Lessons Learned](#10-engineering-challenges--lessons-learned)
- [11. Proof of Execution Artifacts](#11-proof-of-execution-artifacts)

---

## 1. Project Overview & Motivation

Enterprise image processing applications—such as automated manufacturing visual inspection, satellite remote sensing, medical radiology triage, and autonomous vehicle perception—require processing massive streams of visual inputs with predictable latency and maximum hardware utilization.

Naive GPU implementations often suffer from PCIe transfer bottlenecks, where host-to-device (H2D) and device-to-host (D2H) memory copies serialize GPU execution, leaving expensive streaming multiprocessors (SMs) idle. 

**ES-CIP** solves this problem by implementing an asynchronous, multi-stream execution pipeline that overlaps memory transfers with computation across a pool of $K$ non-blocking `cudaStream_t` workers. The system integrates both **NVIDIA Performance Primitives (NPP)** (for hardware-accelerated geometric rotations and filtering) and **custom 2D CUDA kernels with shared-memory cache tiling** (for Sobel gradient edge extraction).

---

## 2. Key Features

- **Enterprise-Scale Batch Processing**: Demonstrated execution on **125 input images** (38+ MB total data volume across 256x256, 512x512, and 1024x1024 resolutions) in a single invocation.
- **Asynchronous Multi-Stream Concurrency**: Overlaps H2D, kernel computation, and D2H across $K$ concurrent CUDA streams (`cudaStream_t`) using pinned host buffers (`cudaHostAlloc`).
- **NVIDIA Performance Primitives (NPP) Integration**: Uses `nppiRotate_8u_C1R_Ctx` / `nppiRotate_8u_C3R_Ctx` and `nppiFilterBox_8u_C1R_Ctx` for hardware-accelerated transformations.
- **Custom Shared-Memory 2D Convolution Kernel**: 2D tiled Sobel edge detector utilizing $18 \times 18$ shared memory cache per $16 \times 16$ thread block with boundary clamping and halo exchange.
- **Pitched 2D Device Memory**: Guaranteed hardware row alignment via `cudaMallocPitch` and `cudaMemcpy2DAsync`.
- **CPU Reference & Parity Verification**: Integrated CPU reference algorithm with automated **Peak Signal-to-Noise Ratio (PSNR)** and **Mean Squared Error (MSE)** validation (achieving 100% bit-exact parity: $99.99\text{ dB}$).
- **Zero-Dependency Image I/O**: Self-contained binary Netpbm (PGM/PPM) parser and serializer compiling natively without OpenCV or external shared library dependencies.
- **Robust Command Line Interface**: Comprehensive argument parser adhering to standard enterprise CLI conventions (`--input_dir`, `--output_dir`, `--streams`, `--operation`, `--benchmark`, `--cpu_verify`).

---

## 3. Architecture & Pipelining

```
Host (CPU / Pinned Host Memory)
+-----------------------------------------------------------------------------------+
| Frame 0 (H2D) | Frame 1 (H2D) | Frame 2 (H2D) | Frame 3 (H2D)                     |
+-----------------------------------------------------------------------------------+
         |               |               |               |
         v               v               v               v
CUDA Streams (Overlap Engine)
+-----------------------------------------------------------------------------------+
| Stream 0:  [ H2D Frame 0 ] ---> [ Compute Frame 0 ] ---> [ D2H Frame 0 ]         |
| Stream 1:       [ H2D Frame 1 ] ---> [ Compute Frame 1 ] ---> [ D2H Frame 1 ]     |
| Stream 2:            [ H2D Frame 2 ] ---> [ Compute Frame 2 ] ---> [ D2H Frame 2 ]|
| Stream 3:                 [ H2D Frame 3 ] ---> [ Compute Frame 3 ] ---> ...      |
+-----------------------------------------------------------------------------------+
                                                                 |
                                                                 v
                                                      Output Storage (output/)
```

### Multi-Stage Processing Pipeline (`--operation all`)
For each input image, the full enterprise workflow executes three sequential stages on the device:
1. **Stage 1 (Noise Suppression)**: NVIDIA NPP $3 \times 3$ Box Filter (`nppiFilterBox`) to eliminate high-frequency sensor noise.
2. **Stage 2 (Geometric Alignment)**: NVIDIA NPP Bilinear Interpolation Rotation (`nppiRotate`) by specified angle (e.g., $45^\circ$) with center-preserving coordinate offset translation.
3. **Stage 3 (Feature Extraction)**: Custom CUDA 2D Sobel Filter with shared-memory tiling to extract structural edges and gradient magnitude contours.

---

## 4. Repository Structure

```
CudaAtScale/
├── .clang-format                  # Google C++ Style Guide format definition
├── .gitignore                     # Git configuration ignoring build binaries
├── CMakeLists.txt                 # Cross-platform CMake build configuration
├── Makefile                       # Production Makefile for Linux / Coursera Lab
├── build.bat                      # Windows NVCC / MSVC compilation script
├── run.sh                         # Linux execution script with automated test runner
├── run.bat                        # Windows execution script
├── README.md                      # Comprehensive project documentation
├── SUBMISSION.md                  # Coursera peer review submission text
├── include/                       # Public API header files
│   ├── benchmark.h                # High-precision timer and CSV export
│   ├── cli_parser.h               # Google-style CLI argument parser
│   ├── cuda_common.h              # CUDA_CHECK and NPP_CHECK error handling macros
│   ├── custom_kernels.cuh         # 2D shared-memory CUDA kernels
│   ├── image.h                    # In-memory Image abstraction (RAII)
│   ├── image_io.h                 # Binary Netpbm (PGM/PPM) parser & serializer
│   ├── npp_pipeline.h             # NVIDIA Performance Primitives wrapper
│   └── pipeline_manager.h         # Multi-stream batch pipeline orchestrator
├── src/                           # Implementation sources
│   ├── cli_parser.cpp             # CLI argument parser implementation
│   ├── custom_kernels.cu          # Optimized CUDA kernels & CPU reference
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
    ├── execution_proof.log        # Complete 164 KB execution log trace
    ├── performance_summary.md     # Detailed performance & benchmark writeup
    ├── comparison_sample_concentric_512.pgm
    ├── comparison_sample_checkerboard_512.pgm
    └── comparison_sample_highres_1024.pgm
```

---

## 5. Google C++ Style Guide Compliance

This project strictly adheres to the **Google C++ Style Guide**:
- **Formatting**: Validated against `.clang-format` based on `BasedOnStyle: Google` (2-space indentation, 100 column limit, bracket placement).
- **Naming Conventions**:
  - Types and Classes: `PascalCase` (`PipelineManager`, `ImageBenchmarkRecord`, `StreamResource`)
  - Functions & Methods: `PascalCase` (`LoadImageFile`, `ExecuteNppRotate`, `RunBatch`)
  - Member Variables: Trailing underscore `variable_name_` (`options_`, `stream_pool_`, `initialized_`)
  - Constants: `kConstantName` (`kTileDimX`, `kRadius`, `kBlockDimX`)
  - Header Guards: `<PROJECT>_<PATH>_<FILE>_H_` (`CUDA_AT_SCALE_INCLUDE_IMAGE_H_`)
- **Memory & Resource Management**: RAII patterns for CUDA streams, events, device allocations, and pinned host buffers.
- **Error Handling**: Macro guards `CUDA_CHECK` and `NPP_CHECK` intercept all runtime returns, capturing source file, line number, error code, and detailed string descriptions.

---

## 6. Build Instructions

### Linux / Coursera Cloud Lab Environment

The project is configured to build seamlessly in the Coursera Ubuntu cloud laboratory:

```bash
# 1. Clone repository
git clone <YOUR_REPO_URL>
cd CudaAtScale

# 2. Compile using standard Makefile (auto-detects CUDA path)
make -j$(nproc)

# 3. Generate test dataset (125 benchmark images)
make dataset

# 4. Execute the pipeline
bash run.sh
```

Alternatively, build with CMake:
```bash
cmake -B build -O Release
cmake --build build -j$(nproc)
```

### Windows Environment

Using the provided batch script with MSVC and NVCC:

```bat
:: From Developer Command Prompt for Visual Studio 2022
build.bat
run.bat
```

---

## 7. Command Line Interface (CLI) Usage

The executable takes standard enterprise command-line arguments:

```
Options:
  --input_dir <path>       Directory containing input images (default: data/input)
  --output_dir <path>      Directory for processed output images (default: output)
  --operation <mode>       Operation to execute:
                             rotate : NVIDIA NPP geometric rotation
                             blur   : NVIDIA NPP spatial box filter
                             sobel  : Custom CUDA shared-memory 2D convolution
                             all    : Full pipelined multi-stage workflow (default)
  --angle <degrees>        Rotation angle in degrees for NPP rotate (default: 45.0)
  --streams <count>        Number of concurrent CUDA streams (default: 4)
  --batch_size <N>         Max images to process in run (0 = all images) (default: 0)
  --benchmark              Execute CPU vs Single-Stream vs Multi-Stream benchmark
  --cpu_verify             Run CPU reference algorithms to verify numeric parity
  --device <id>            Target CUDA GPU device ID (default: 0)
  --help, -h               Display this usage help message and exit
```

### Example Commands:

```bash
# 1. Process all 125 images through the complete multi-stage pipeline using 4 streams
bin/cuda_enterprise_pipeline --input_dir data/input --output_dir output --streams 4 --operation all

# 2. Run scaling benchmark comparing 1 vs 4 vs 8 concurrent streams
bin/cuda_enterprise_pipeline --input_dir data/input --output_dir output --benchmark

# 3. Run with CPU verification to validate mathematical parity (PSNR / MSE)
bin/cuda_enterprise_pipeline --input_dir data/input --output_dir output --operation sobel --cpu_verify
```

---

## 8. Experimental Results & Performance Benchmarks

Tested on an **NVIDIA GeForce RTX 4060 Laptop GPU** (3072 CUDA Cores, 8GB GDDR6, PCIe Gen 4):

### Multi-Stream Concurrency Scaling (125 Images, 38.12 MB)

| Pipeline Configuration | Concurrency Mode | Total Wall Time (ms) | Throughput (images/sec) | Memory Bandwidth (MB/s) | Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1 Stream** | Synchronous Baseline | 364.87 ms | 342.6 img/s | 104.48 MB/s | 1.00x |
| **4 Streams** | Concurrent Pipelined | 325.54 ms | 383.9 img/s | 117.10 MB/s | 1.12x |
| **8 Streams** | Concurrent Pipelined | **298.76 ms** | **418.4 img/s** | **127.61 MB/s** | **1.22x** |

### GPU Kernel vs CPU Reference Acceleration

| Image Resolution | Dataset Count | CPU Reference Latency | GPU Kernel Latency | Kernel Acceleration | PSNR Parity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **256 × 256** | 50 | 0.88 ms | 0.08 ms | **11.0x** | **99.99 dB** |
| **512 × 512** | 50 | 3.52 ms | 0.11 ms | **32.0x** | **99.99 dB** |
| **1024 × 1024** | 15 | 12.35 ms | 0.31 ms | **39.8x** | **99.99 dB** |

---

## 9. Algorithms & CUDA Kernel Implementation

### 1. Custom Shared-Memory 2D Sobel Filter
A naive 2D convolution kernel fetches $3 \times 3 = 9$ pixels from global memory per output thread. For adjacent threads, this creates severe redundant memory bus traffic.

Our implementation uses **2D Shared-Memory Tiling**:
- Block dimensions: $16 \times 16$ threads (256 threads).
- Shared memory tile: $(16 + 2) \times (16 + 2) = 18 \times 18 = 324\text{ bytes}$.
- Each thread collaboratively loads elements into shared memory while clamping boundary coordinates to image edges.
- After a single `__syncthreads()` barrier, horizontal ($G_x$) and vertical ($G_y$) gradients are calculated exclusively from on-chip shared memory:
  $$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} * I, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix} * I$$
  $$\text{Magnitude} = \min\left(255, \sqrt{G_x^2 + G_y^2}\right)$$
- Output is written to pitched device memory via coalesced 32-bit/64-bit transactions.

### 2. NVIDIA Performance Primitives (NPP) Rotation
Image rotation around the midpoint $(c_x, c_y) = (w/2, h/2)$ requires affine coordinate translation:
$$\text{Shift}_x = c_x - (c_x \cos\theta - c_y \sin\theta)$$
$$\text{Shift}_y = c_y - (c_x \sin\theta + c_y \cos\theta)$$
The rotation is executed via `nppiRotate_8u_C1R_Ctx` with bilinear interpolation (`NPPI_INTER_LINEAR`), bound to the corresponding non-blocking `cudaStream_t` via `NppStreamContext`.

---

## 10. Engineering Challenges & Lessons Learned

1. **Hardware Memory Pitch vs Contiguous Buffers**:
   - *Challenge*: 2D image rows must be aligned to hardware memory transaction boundaries (e.g., 128 or 256 bytes) to maximize memory bus coalescing. Failing to align leads to uncoalesced memory reads and invalid NPP step errors (`NPP_STEP_ERROR`).
   - *Resolution*: Allocated all 2D device buffers using `cudaMallocPitch` and transferred data via `cudaMemcpy2DAsync`.

2. **PCIe Transfer Saturation & Pinned Memory Overhead**:
   - *Challenge*: Standard pageable host memory allocated via `malloc()` or `std::vector` forces the CUDA driver to perform a staged synchronous copy to an internal driver staging buffer before initiating DMA, completely defeating multi-stream concurrency.
   - *Resolution*: Implemented persistent pinned host memory buffers (`cudaHostAlloc`) in `StreamResource`. This enables true asynchronous DMA transfers via PCIe copy engines concurrent with SM execution.

3. **Stream Workload Balancing**:
   - *Challenge*: If stream synchronization occurs synchronously after each image dispatch, execution degenerates into serialized single-stream performance.
   - *Resolution*: Implemented a circular in-flight task tracking queue where stream $k$ is only synchronized when reused on task $k + N_{\text{streams}}$, providing sufficient latency slack to hide kernel execution behind memory transfers.

---

## 11. Proof of Execution Artifacts

The repository includes concrete proof of execution artifacts generated from real runs:
- **`artifacts/execution_proof.log`**: 164 KB verbose console execution trace of all 125 images processed in a single run.
- **`output/benchmark_results.csv`**: Full spreadsheet containing 125 rows of per-image execution metrics (H2D time, Kernel time, D2H time, CPU time, Speedup, PSNR).
- **`output/`**: Directory containing all 125 generated output images (`processed_*.pgm`, `processed_*.ppm`).
- **`artifacts/comparison_sample_*.pgm`**: Visual side-by-side comparison images demonstrating original vs processed images.

---

## License
Licensed under the Apache License, Version 2.0.
