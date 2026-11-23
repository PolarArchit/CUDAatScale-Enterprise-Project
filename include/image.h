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

#ifndef CUDA_AT_SCALE_INCLUDE_IMAGE_H_
#define CUDA_AT_SCALE_INCLUDE_IMAGE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cuda_scale {

// Supported image formats.
enum class ImageFormat {
  kGrayscale = 1,  // 1 byte per pixel (PGM P5)
  kRGB = 3         // 3 bytes per pixel (PPM P6)
};

// Represents an in-memory 2D image buffer.
struct Image {
  int width = 0;
  int height = 0;
  int channels = 1;
  std::vector<uint8_t> data;

  Image() = default;

  Image(int w, int h, int c = 1)
      : width(w), height(h), channels(c), data(static_cast<size_t>(w) * h * c, 0) {}

  Image(int w, int h, int c, const uint8_t* raw_data)
      : width(w), height(h), channels(c),
        data(raw_data, raw_data + (static_cast<size_t>(w) * h * c)) {}

  // Total byte size of the image buffer.
  size_t SizeInBytes() const {
    return static_cast<size_t>(width) * height * channels;
  }

  // Row step / pitch in bytes (contiguous host layout).
  int Pitch() const {
    return width * channels;
  }

  // Check if image dimensions are positive and data buffer matches.
  bool IsValid() const {
    return width > 0 && height > 0 && channels > 0 &&
           data.size() == SizeInBytes();
  }

  // Resize and zero-initialize image buffer.
  void Allocate(int w, int h, int c = 1) {
    width = w;
    height = h;
    channels = c;
    data.assign(static_cast<size_t>(w) * h * c, 0);
  }

  // Access pixel byte at (x, y, channel).
  uint8_t& At(int x, int y, int c = 0) {
    return data[(static_cast<size_t>(y) * width + x) * channels + c];
  }

  const uint8_t& At(int x, int y, int c = 0) const {
    return data[(static_cast<size_t>(y) * width + x) * channels + c];
  }
};

}  // namespace cuda_scale

#endif  // CUDA_AT_SCALE_INCLUDE_IMAGE_H_
