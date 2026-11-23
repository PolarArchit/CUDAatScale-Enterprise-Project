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

#ifndef CUDA_AT_SCALE_INCLUDE_IMAGE_IO_H_
#define CUDA_AT_SCALE_INCLUDE_IMAGE_IO_H_

#include <string>
#include <vector>

#include "image.h"

namespace cuda_scale {

// Reads a PGM (P5 Grayscale binary) image from file.
bool ReadPGM(const std::string& filepath, Image* image);

// Writes a PGM (P5 Grayscale binary) image to file.
bool WritePGM(const std::string& filepath, const Image& image);

// Reads a PPM (P6 RGB binary) image from file.
bool ReadPPM(const std::string& filepath, Image* image);

// Writes a PPM (P6 RGB binary) image to file.
bool WritePPM(const std::string& filepath, const Image& image);

// Loads an image file, automatically detecting format by header or extension.
bool LoadImageFile(const std::string& filepath, Image* image);

// Saves an image file, choosing format based on channel count and extension.
bool SaveImageFile(const std::string& filepath, const Image& image);

// Enumerates all image files (.pgm, .ppm) in the specified directory.
std::vector<std::string> GetImageFilesInDirectory(const std::string& dir_path);

}  // namespace cuda_scale

#endif  // CUDA_AT_SCALE_INCLUDE_IMAGE_IO_H_
