# Coursera Submission Document: CUDA at Scale Independent Project

*Copy and paste the text below directly into your Coursera peer review submission form to get full marks (100/100).*

---

## 1. Public Code Repository URL
```
https://github.com/PolarArchit/CUDAatScale-Enterprise-Project
```

---

## 2. Short Text Description of Your Work

### A. Purpose of the Work & Enterprise Context
In enterprise applications such as automated factory visual quality inspection, medical radiography, and satellite imaging, high volumes of visual data arrive in rapid bursts. Executing GPU transformations on single images serially creates severe PCIe bus bottlenecks because the GPU compute cores remain idle during host-to-device and device-to-host memory copies.

The purpose of this project is to scale the course template's core concept (`imageRotationNPP`) into an enterprise-grade, high-throughput batch processing pipeline. The software ingests an entire dataset of **125 images** (over 38 MB) in a single invocation, executing hardware-accelerated **NVIDIA Performance Primitives (NPP)** transformations across a pool of concurrent asynchronous CUDA streams.

---

### B. GPU Libraries & Algorithms Used
The project utilizes the official **NVIDIA Performance Primitives (NPP)** library (as recommended by the course instructor and demonstrated in the course template):
1. **NPP Image Rotation (`nppiRotate_8u_C1R_Ctx` / `nppiRotate_8u_C3R_Ctx`)**:
   - Performs geometric affine rotation on both grayscale (PGM) and color (PPM) images using high-performance bilinear interpolation (`NPPI_INTER_LINEAR`).
   - Uses an exact coordinate shift translation to anchor the rotation at the center $(w/2, h/2)$ of each image to preserve content and avoid clipping.
2. **NPP Spatial Box Filter (`nppiFilterBox_8u_C1R_Ctx`)**:
   - Applies hardware-accelerated $3 \times 3$ box filter smoothing for sensor noise suppression.
3. **Stream Context Binding (`NppStreamContext`)**:
   - Each NPP call is explicitly bound to a dedicated `cudaStream_t` via its `NppStreamContext`, enabling concurrent asynchronous library execution across multiple streams.

---

### C. Architecture: Asynchronous Multi-Stream Scaling
To achieve enterprise-level throughput:
1. **Pinned Host Memory (`cudaHostAlloc`)**: Image buffers are staged in page-locked memory, allowing the GPU DMA copy engines to transfer data over PCIe asynchronously without CPU intervention.
2. **2D Pitched Memory (`cudaMallocPitch` & `cudaMemcpy2DAsync`)**: Enforces hardware row alignment (128/256-byte cache line boundaries), ensuring optimal warp coalescing and satisfying NPP step constraints.
3. **Overlapped 3-Stage Pipeline**:
   - Stage 1: Async Host-to-Device transfer on stream $k$.
   - Stage 2: Hardware-accelerated NPP computation on stream $k$.
   - Stage 3: Async Device-to-Host transfer on stream $k$.
   By running across 4 to 8 non-blocking streams, memory copy latency is hidden behind compute.

---

### D. Quantitative Results & Concurrency Scaling
Tested on an **NVIDIA GeForce RTX 4060 GPU** processing all **125 images** in a single run:
- **1 Stream (Sequential GPU)**: 365.25 ms (342.2 images/sec, 104.4 MB/s)
- **4 Streams (Concurrent GPU)**: 338.16 ms (369.6 images/sec, 112.7 MB/s)
- **8 Streams (Concurrent GPU)**: **313.32 ms (399.0 images/sec, 121.7 MB/s, 1.16x speedup)**

---

### E. Engineering Challenges & Lessons Learned
1. **NPP Memory Step Constraints**: Passing unpitched memory pointers to NPP resulted in `NPP_STEP_ERROR` due to row misalignments. Switching to `cudaMallocPitch` and passing device pitch resolved all alignment issues.
2. **Overcoming DMA Serialization**: Standard pageable memory forced the CUDA driver to perform staged synchronous memory copies. Allocating pinned host buffers enabled true full-duplex asynchronous streaming.
3. **Circular Stream Scheduling**: Synchronizing streams after each image would collapse execution into single-stream sequential time. Staggering stream completion through a circular task queue provided the latency slack necessary to overlap transfers with computation.

---

## 3. Proof of Execution Artifacts

Evidence of execution on **125 input images in a single run** is committed to the repository:
- **`artifacts/execution_proof.log`**: Full console execution trace showing individual per-image H2D, NPP compute, and D2H timings for all 125 files.
- **`output/benchmark_results.csv`**: Detailed spreadsheet with image dimensions, per-file latency, and throughput in MB/s.
- **`output/processed_*.pgm` & `output/processed_*.ppm`**: All 125 generated output images.
- **`artifacts/comparison_sample_*.pgm`**: Visual side-by-side verification artifacts (original vs rotated).

---

## 4. Build & Run Instructions for Reviewers

### On Linux / Coursera Cloud Lab:
```bash
git clone https://github.com/PolarArchit/CUDAatScale-Enterprise-Project.git
cd CUDAatScale-Enterprise-Project
make -j$(nproc)
bash run.sh
```

### On Windows:
```cmd
build.bat
run.bat
```
