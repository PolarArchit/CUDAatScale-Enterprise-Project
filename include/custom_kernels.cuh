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

#ifndef CUDA_AT_SCALE_INCLUDE_CUSTOM_KERNELS_CUH_
#define CUDA_AT_SCALE_INCLUDE_CUSTOM_KERNELS_CUH_

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

#include "image.h"

namespace cuda_scale {

// Launches the optimized 2D Sobel edge detection kernel using shared memory tiling.
// Works on single-channel grayscale 8-bit images.
void LaunchGpuSobelFilter(const uint8_t* d_input, size_t in_pitch,
                          uint8_t* d_output, size_t out_pitch,
                          int width, int height,
                          cudaStream_t stream = nullptr);

// Launches the optimized 2D Box blur kernel using shared memory tiling.
void LaunchGpuBoxFilter(const uint8_t* d_input, size_t in_pitch,
                        uint8_t* d_output, size_t out_pitch,
                        int width, int height,
                        cudaStream_t stream = nullptr);

// Sequential CPU reference implementation of Sobel filter for benchmarking and verification.
void CpuSobelFilter(const Image& input, Image* output);

// Sequential CPU reference implementation of Box blur for benchmarking and verification.
void CpuBoxFilter(const Image& input, Image* output);

// Computes Mean Squared Error (MSE) between two images.
double ComputeMSE(const Image& img1, const Image& img2);

// Computes Peak Signal-to-Noise Ratio (PSNR) in dB between two images.
double ComputePSNR(const Image& img1, const Image& img2);

}  // namespace cuda_scale

#endif  // CUDA_AT_SCALE_INCLUDE_CUSTOM_KERNELS_CUH_
