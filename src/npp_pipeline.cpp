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

#include "npp_pipeline.h"

#include <cuda_runtime.h>
#include <npp.h>

#include <cmath>
#include <iostream>

#include "cuda_common.h"

namespace cuda_scale {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Helper to configure NPP stream context.
NppStreamContext GetConfiguredStreamContext(cudaStream_t stream) {
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  if (stream != nullptr) {
    ctx.hStream = stream;
  }
  return ctx;
}

}  // namespace

bool ExecuteNppRotate(const uint8_t* d_src, size_t src_pitch,
                     uint8_t* d_dst, size_t dst_pitch,
                     int width, int height, int channels,
                     double angle_degrees,
                     cudaStream_t stream) {
  if (!d_src || !d_dst || width <= 0 || height <= 0) {
    return false;
  }

  // Clear destination buffer to black (0) for border padding.
  CUDA_CHECK(cudaMemset2DAsync(d_dst, dst_pitch, 0,
                               width * channels, height, stream));

  NppiSize oSrcSize = {width, height};
  NppiRect oSrcROI = {0, 0, width, height};
  NppiRect oDstROI = {0, 0, width, height};

  // Compute shift to center rotation at image center (w/2, h/2).
  const double rad = angle_degrees * (kPi / 180.0);
  const double cx = width / 2.0;
  const double cy = height / 2.0;
  const double cos_a = std::cos(rad);
  const double sin_a = std::sin(rad);

  const double shift_x = cx - (cx * cos_a - cy * sin_a);
  const double shift_y = cy - (cx * sin_a + cy * cos_a);

  NppStreamContext ctx = GetConfiguredStreamContext(stream);

  NppStatus status = NPP_SUCCESS;
  if (channels == 1) {
    status = nppiRotate_8u_C1R_Ctx(d_src, oSrcSize, static_cast<int>(src_pitch),
                                   oSrcROI, d_dst, static_cast<int>(dst_pitch),
                                   oDstROI, angle_degrees, shift_x, shift_y,
                                   NPPI_INTER_LINEAR, ctx);
  } else if (channels == 3) {
    status = nppiRotate_8u_C3R_Ctx(d_src, oSrcSize, static_cast<int>(src_pitch),
                                   oSrcROI, d_dst, static_cast<int>(dst_pitch),
                                   oDstROI, angle_degrees, shift_x, shift_y,
                                   NPPI_INTER_LINEAR, ctx);
  } else {
    std::cerr << "[NPP] Unsupported channel count for rotation: " << channels
              << std::endl;
    return false;
  }

  if (status != NPP_SUCCESS) {
    std::cerr << "[NPP] Error during nppiRotate: " << NppGetErrorString(status)
              << std::endl;
    return false;
  }

  return true;
}

bool ExecuteNppBoxFilter(const uint8_t* d_src, size_t src_pitch,
                        uint8_t* d_dst, size_t dst_pitch,
                        int width, int height, int channels,
                        int mask_size,
                        cudaStream_t stream) {
  if (!d_src || !d_dst || width <= 0 || height <= 0) {
    return false;
  }

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {mask_size, mask_size};
  NppiPoint oAnchor = {mask_size / 2, mask_size / 2};

  NppStreamContext ctx = GetConfiguredStreamContext(stream);

  NppStatus status = NPP_SUCCESS;
  if (channels == 1) {
    status = nppiFilterBox_8u_C1R_Ctx(d_src, static_cast<int>(src_pitch),
                                      d_dst, static_cast<int>(dst_pitch),
                                      oSizeROI, oMaskSize, oAnchor, ctx);
  } else if (channels == 3) {
    status = nppiFilterBox_8u_C3R_Ctx(d_src, static_cast<int>(src_pitch),
                                      d_dst, static_cast<int>(dst_pitch),
                                      oSizeROI, oMaskSize, oAnchor, ctx);
  } else {
    std::cerr << "[NPP] Unsupported channel count for box filter: " << channels
              << std::endl;
    return false;
  }

  if (status != NPP_SUCCESS) {
    std::cerr << "[NPP] Error during nppiFilterBox: " << NppGetErrorString(status)
              << std::endl;
    return false;
  }

  return true;
}

}  // namespace cuda_scale
