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

#ifndef CUDA_AT_SCALE_INCLUDE_NPP_PIPELINE_H_
#define CUDA_AT_SCALE_INCLUDE_NPP_PIPELINE_H_

#include <cuda_runtime.h>
#include <npp.h>

#include <cstdint>

#include "image.h"

namespace cuda_scale {

// Executes NVIDIA NPP image rotation on pitched GPU memory.
// Supports 1-channel (grayscale) and 3-channel (RGB) images.
bool ExecuteNppRotate(const uint8_t* d_src, size_t src_pitch,
                     uint8_t* d_dst, size_t dst_pitch,
                     int width, int height, int channels,
                     double angle_degrees,
                     cudaStream_t stream = nullptr);

// Executes NVIDIA NPP spatial box filter smoothing on pitched GPU memory.
bool ExecuteNppBoxFilter(const uint8_t* d_src, size_t src_pitch,
                        uint8_t* d_dst, size_t dst_pitch,
                        int width, int height, int channels,
                        int mask_size = 3,
                        cudaStream_t stream = nullptr);

}  // namespace cuda_scale

#endif  // CUDA_AT_SCALE_INCLUDE_NPP_PIPELINE_H_
