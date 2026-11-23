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

#include "custom_kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>

#include "cuda_common.h"

namespace cuda_scale {

// Constants for shared-memory tile dimensions.
constexpr int kBlockDimX = 16;
constexpr int kBlockDimY = 16;
constexpr int kRadius = 1;
constexpr int kTileDimX = kBlockDimX + 2 * kRadius;  // 18
constexpr int kTileDimY = kBlockDimY + 2 * kRadius;  // 18

// Device helper function to clamp coordinates to valid image boundaries.
__device__ inline int DeviceClamp(int val, int min_val, int max_val) {
  return max(min_val, min(val, max_val));
}

// -----------------------------------------------------------------------------
// CUDA Kernel: Shared-Memory Tiled Sobel 2D Edge Detection
// -----------------------------------------------------------------------------
// Uses an 18x18 shared memory cache per 16x16 thread block to load data and
// halos in parallel, minimizing DRAM bandwidth and memory bus contention.
__global__ void SobelFilterKernel(const uint8_t* __restrict__ d_input,
                                  size_t in_pitch,
                                  uint8_t* __restrict__ d_output,
                                  size_t out_pitch,
                                  int width, int height) {
  // Shared memory tile caching current block pixels plus 1-pixel halo.
  __shared__ uint8_t tile[kTileDimY][kTileDimX];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int gx = blockIdx.x * blockDim.x + tx;
  const int gy = blockIdx.y * blockDim.y + ty;

  // Linear thread ID in block [0 .. 255]
  const int tid = ty * blockDim.x + tx;
  const int num_threads = blockDim.x * blockDim.y;  // 256
  const int total_tile_elements = kTileDimX * kTileDimY;  // 324

  // Top-left origin of this tile in global image coordinates.
  const int tile_origin_x = blockIdx.x * blockDim.x - kRadius;
  const int tile_origin_y = blockIdx.y * blockDim.y - kRadius;

  // Collaboratively load all 324 shared memory elements.
  for (int idx = tid; idx < total_tile_elements; idx += num_threads) {
    const int tile_y = idx / kTileDimX;
    const int tile_x = idx % kTileDimX;

    const int img_x = DeviceClamp(tile_origin_x + tile_x, 0, width - 1);
    const int img_y = DeviceClamp(tile_origin_y + tile_y, 0, height - 1);

    const uint8_t* row_ptr = d_input + img_y * in_pitch;
    tile[tile_y][tile_x] = row_ptr[img_x];
  }

  // Ensure all threads have populated shared memory before computation.
  __syncthreads();

  // If outside actual image dimensions, do not write output.
  if (gx >= width || gy >= height) {
    return;
  }

  // Local tile coordinates corresponding to (gx, gy).
  const int lx = tx + kRadius;
  const int ly = ty + kRadius;

  // 3x3 neighborhood values cached in shared memory.
  const int p00 = tile[ly - 1][lx - 1];
  const int p01 = tile[ly - 1][lx];
  const int p02 = tile[ly - 1][lx + 1];

  const int p10 = tile[ly][lx - 1];
  const int p12 = tile[ly][lx + 1];

  const int p20 = tile[ly + 1][lx - 1];
  const int p21 = tile[ly + 1][lx];
  const int p22 = tile[ly + 1][lx + 1];

  // Horizontal Sobel gradient Gx:
  // [-1  0  1]
  // [-2  0  2]
  // [-1  0  1]
  const int gx_grad = (-p00 + p02) + 2 * (-p10 + p12) + (-p20 + p22);

  // Vertical Sobel gradient Gy:
  // [-1 -2 -1]
  // [ 0  0  0]
  // [ 1  2  1]
  const int gy_grad = (-p00 - 2 * p01 - p02) + (p20 + 2 * p21 + p22);

  // Gradient magnitude approximation or Euclidean norm.
  const float mag = sqrtf(static_cast<float>(gx_grad * gx_grad + gy_grad * gy_grad));
  const uint8_t result = static_cast<uint8_t>(fminf(255.0f, mag));

  // Coalesced write to pitched output memory.
  uint8_t* out_row = d_output + gy * out_pitch;
  out_row[gx] = result;
}

// -----------------------------------------------------------------------------
// CUDA Kernel: Shared-Memory Tiled Box Blur (3x3 spatial filter)
// -----------------------------------------------------------------------------
__global__ void BoxFilterKernel(const uint8_t* __restrict__ d_input,
                                size_t in_pitch,
                                uint8_t* __restrict__ d_output,
                                size_t out_pitch,
                                int width, int height) {
  __shared__ uint8_t tile[kTileDimY][kTileDimX];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int gx = blockIdx.x * blockDim.x + tx;
  const int gy = blockIdx.y * blockDim.y + ty;

  const int tid = ty * blockDim.x + tx;
  const int num_threads = blockDim.x * blockDim.y;
  const int total_tile_elements = kTileDimX * kTileDimY;

  const int tile_origin_x = blockIdx.x * blockDim.x - kRadius;
  const int tile_origin_y = blockIdx.y * blockDim.y - kRadius;

  for (int idx = tid; idx < total_tile_elements; idx += num_threads) {
    const int tile_y = idx / kTileDimX;
    const int tile_x = idx % kTileDimX;

    const int img_x = DeviceClamp(tile_origin_x + tile_x, 0, width - 1);
    const int img_y = DeviceClamp(tile_origin_y + tile_y, 0, height - 1);

    const uint8_t* row_ptr = d_input + img_y * in_pitch;
    tile[tile_y][tile_x] = row_ptr[img_x];
  }

  __syncthreads();

  if (gx >= width || gy >= height) {
    return;
  }

  const int lx = tx + kRadius;
  const int ly = ty + kRadius;

  int sum = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      sum += tile[ly + dy][lx + dx];
    }
  }

  uint8_t* out_row = d_output + gy * out_pitch;
  out_row[gx] = static_cast<uint8_t>(sum / 9);
}

void LaunchGpuSobelFilter(const uint8_t* d_input, size_t in_pitch,
                          uint8_t* d_output, size_t out_pitch,
                          int width, int height,
                          cudaStream_t stream) {
  dim3 block(kBlockDimX, kBlockDimY);
  dim3 grid((width + kBlockDimX - 1) / kBlockDimX,
            (height + kBlockDimY - 1) / kBlockDimY);

  SobelFilterKernel<<<grid, block, 0, stream>>>(d_input, in_pitch, d_output,
                                                out_pitch, width, height);
  CUDA_CHECK(cudaGetLastError());
}

void LaunchGpuBoxFilter(const uint8_t* d_input, size_t in_pitch,
                        uint8_t* d_output, size_t out_pitch,
                        int width, int height,
                        cudaStream_t stream) {
  dim3 block(kBlockDimX, kBlockDimY);
  dim3 grid((width + kBlockDimX - 1) / kBlockDimX,
            (height + kBlockDimY - 1) / kBlockDimY);

  BoxFilterKernel<<<grid, block, 0, stream>>>(d_input, in_pitch, d_output,
                                              out_pitch, width, height);
  CUDA_CHECK(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// CPU Sequential Reference Implementations
// -----------------------------------------------------------------------------
void CpuSobelFilter(const Image& input, Image* output) {
  if (!output) return;
  const int w = input.width;
  const int h = input.height;
  output->Allocate(w, h, 1);

  auto clamp_val = [](int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
  };

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int p00 = input.At(clamp_val(x - 1, 0, w - 1), clamp_val(y - 1, 0, h - 1));
      int p01 = input.At(x, clamp_val(y - 1, 0, h - 1));
      int p02 = input.At(clamp_val(x + 1, 0, w - 1), clamp_val(y - 1, 0, h - 1));

      int p10 = input.At(clamp_val(x - 1, 0, w - 1), y);
      int p12 = input.At(clamp_val(x + 1, 0, w - 1), y);

      int p20 = input.At(clamp_val(x - 1, 0, w - 1), clamp_val(y + 1, 0, h - 1));
      int p21 = input.At(x, clamp_val(y + 1, 0, h - 1));
      int p22 = input.At(clamp_val(x + 1, 0, w - 1), clamp_val(y + 1, 0, h - 1));

      int gx = (-p00 + p02) + 2 * (-p10 + p12) + (-p20 + p22);
      int gy = (-p00 - 2 * p01 - p02) + (p20 + 2 * p21 + p22);

      float mag = std::sqrt(static_cast<float>(gx * gx + gy * gy));
      output->At(x, y) = static_cast<uint8_t>(std::min(255.0f, mag));
    }
  }
}

void CpuBoxFilter(const Image& input, Image* output) {
  if (!output) return;
  const int w = input.width;
  const int h = input.height;
  output->Allocate(w, h, 1);

  auto clamp_val = [](int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
  };

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int sum = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          sum += input.At(clamp_val(x + dx, 0, w - 1),
                          clamp_val(y + dy, 0, h - 1));
        }
      }
      output->At(x, y) = static_cast<uint8_t>(sum / 9);
    }
  }
}

double ComputeMSE(const Image& img1, const Image& img2) {
  if (img1.width != img2.width || img1.height != img2.height ||
      img1.channels != img2.channels) {
    return -1.0;
  }
  double sum_sq = 0.0;
  size_t total_elements = img1.SizeInBytes();
  for (size_t i = 0; i < total_elements; ++i) {
    double diff = static_cast<double>(img1.data[i]) - static_cast<double>(img2.data[i]);
    sum_sq += diff * diff;
  }
  return sum_sq / static_cast<double>(total_elements);
}

double ComputePSNR(const Image& img1, const Image& img2) {
  double mse = ComputeMSE(img1, img2);
  if (mse <= 1e-10) return 99.99;  // Effectively identical
  return 10.0 * std::log10((255.0 * 255.0) / mse);
}

}  // namespace cuda_scale
